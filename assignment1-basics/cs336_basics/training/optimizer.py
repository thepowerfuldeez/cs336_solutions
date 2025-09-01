import math
from typing import Any
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4):
        super().__init__(params=params, defaults=dict(lr=lr))

    def step(self, closure: Any | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 1e-6,
        eps: float = 1e-8,
    ):
        super().__init__(
            params=params, defaults=dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        )

    def step(self, closure: Any | None = None):
        loss = None if closure is None else closure()
        total_update_sq, total_weight_sq = 0.0, 0.0
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                # init first and second order moments if we haven't already
                m: Tensor = state.get("m", torch.zeros_like(p.data, requires_grad=False))
                v: Tensor = state.get("v", torch.zeros_like(p.data, requires_grad=False))

                grad: Tensor = p.grad.data

                m: Tensor = beta1 * m + (1 - beta1) * grad
                v: Tensor = beta2 * v + (1 - beta2) * torch.square(grad)
                bias_correction_term: float = math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # ! wd is decoupled, it should go first !
                wd_delta = -lr * wd * p.data
                adam_delta = -lr * bias_correction_term * m / (torch.sqrt(v) + eps)
                delta = wd_delta + adam_delta

                total_update_sq += (delta.float().norm() ** 2).item()
                total_weight_sq += (p.data.float().norm() ** 2).item()

                p.data.add_(delta)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        update_ratio = math.sqrt(total_update_sq) / (math.sqrt(total_weight_sq) + eps)
        return (loss, update_ratio)


# -----------------------------------------------------------------------------
# Muon optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            momentum = group["momentum"]
            for p in params:
                grad = p.grad

                # out_dim / inp_dim (reversed order bc in our implementation of Linear dims are swapped)
                mup_mult = max(1, p.size(-2) / p.size(-1)) ** 0.5
                # variance preserving multiplier TODO: try on longer runs
                # var_preserving_multiplier = 0.2 * max(p.size(-2), p.size(-1)) ** 0.5
                var_preserving_multiplier = 1.0
                eff_lr = (
                    group["lr"] * mup_mult * var_preserving_multiplier * getattr(p, "lr_mul", 1.0)
                )
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]

                # apply wd
                p.mul_(1 - eff_weight_decay)
                # interpolate momentum
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-eff_lr)


def get_cosine_lr(
    t: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int
) -> float:
    """
    Update learning rate based on cosine schedule with warmup

    t: int - current step
    lr_max: float - max learning rate (usually set as original learning rate)
    lr_min: float - minimum learning rate after decay
    warmup_steps: int - warmup steps starting from ~0 (t / warmup_steps) * lr_max
    cosine_steps: int - total number of steps in the cosine schedule, starting from warmup_steps to cosine steps

    Returns: updated lr
    """
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    elif warmup_steps <= t < cosine_steps:
        cos_lr: float = lr_min + 0.5 * (
            1 + math.cos((t - warmup_steps) / (cosine_steps - warmup_steps) * math.pi)
        ) * (lr_max - lr_min)
        return cos_lr
    else:
        return lr_min


def get_wsd_lr(
    t: int, lr_max: float, lr_min: float, warmup_steps: int, stable_steps: int, decay_steps: int
) -> float:
    """
    Update learning rate based on Warmup Stable Decay schedule

    t: int - current step
    lr_max: float - max learning rate (usually set as original learning rate)
    lr_min: float - minimum learning rate after decay
    warmup_steps: int - warmup steps starting from ~0 (t / warmup_steps) * lr_max
    stable_steps: int - total number of steps in the stable state [warmup_steps, decay_steps]
    decay_steps: int - total number of steps in the decay state to lr_min [decay_steps, total_steps]

    Returns: updated lr
    """
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    elif warmup_steps <= t < warmup_steps + stable_steps:
        return lr_max
    else:
        # t >= warmup_steps + stable_steps
        return (1 - (t - warmup_steps - stable_steps) / decay_steps) * (lr_max - lr_min)


def clip_grad_norm_(
    params: Iterable[nn.Parameter], max_grad_norm: float = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Clips gradients to `max_grad_norm` in-place
    """
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads), "grads are empty!"
    total_squared_norm = torch.zeros((1,), dtype=grads[0].dtype, device=grads[0].device)
    for g in grads:
        total_squared_norm += torch.linalg.norm(g) ** 2
    norm = total_squared_norm.sqrt()
    if norm >= max_grad_norm:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(max_grad_norm / (norm + eps))
    return norm


if __name__ == "__main__":
    weights = nn.Parameter(torch.randn(10, 10))
    opt = SGD([weights], lr=1e0)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).norm()
        print(loss.item())
        loss.backward()
        opt.step()
