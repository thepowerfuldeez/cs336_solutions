import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(logits: Float[Tensor, "... seq vocab_size"], targets: Int[Tensor, "... seq"]) -> Tensor:
    m: Float[Tensor, "... seq 1"] = logits.max(dim=-1, keepdim=True).values
    target_logits: Float[Tensor, "... seq 1"] = logits.gather(dim=-1, index=targets.long().unsqueeze(-1))
    loss: Float[Tensor, "... seq 1"] = m - target_logits + torch.logsumexp(logits - m, dim=-1, keepdim=True)
    return loss.mean()
