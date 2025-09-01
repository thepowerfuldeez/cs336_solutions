import timeit
import json
import argparse
import torch
import numpy as np
import torch.cuda.nvtx as nvtx

from dataclasses import asdict

from cs336_basics.configs.gpt_small_faster import cfg
from cs336_basics.utils.logger import logger
from cs336_basics.utils.config_tools import apply_overrides
from cs336_basics.transformer import Transformer
from cs336_basics.training import cross_entropy, clip_grad_norm_, AdamW, Muon


_SIZES = {
    "small": {"model.d_model": 768, "model.d_ff": 3072, "model.n_layers": 12, "model.n_heads": 12},
    "medium": {"model.d_model": 1024, "model.d_ff": 4096, "model.n_layers": 24, "model.n_heads": 16},
    "large": {"model.d_model": 1280, "model.d_ff": 5120, "model.n_layers": 36, "model.n_heads": 20},
    "xl": {"model.d_model": 1600, "model.d_ff": 6400, "model.n_layers": 48, "model.n_heads": 25},
    "2.7b": {"model.d_model": 2560, "model.d_ff": 10240, "model.n_layers": 32, "model.n_heads": 32},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--override", type=str, help='{"k": "v"} override to the cfg as dict')
    p.add_argument("--forward-only", type=bool, default=False)
    p.add_argument("--warmup-steps", type=int, default=5)
    return p.parse_args()


def init_optimizers(model, d_model):
    one_d_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embedding" in n or "lm_head" in n]
    # Implement muP scaling, for embedding it's sqrt(d), for out it's 0.5
    for n, p in model.named_parameters():
        if "embedding" in n:
            setattr(p, "lr_mul", d_model**0.5)
        elif "lm_head" in n:
            setattr(p, "lr_mul", 0.5)
    two_d_params = [
        p for n, p in model.named_parameters() if p.ndim >= 2 and "embedding" not in n and "lm_head" not in n
    ]
    optimizer1 = AdamW(
        one_d_params,
        lr=1e-4,
        betas=(0.95, 0.99),
        weight_decay=0.0,
    )
    optimizer2 = Muon(
        two_d_params,
        lr=1e-2,
        momentum=0.95,
        weight_decay=1e-4,
    )
    optimizers = [optimizer1, optimizer2]
    return optimizers


def step(model, inputs, optimizers, forward_only: bool = False):
    nvtx.range_push("forward")
    with torch.autocast("cuda", enabled=True):
        logits, prenorm_activation_norms = model(inputs)
        targets = inputs
        loss, z_loss = cross_entropy(logits, targets)
    nvtx.range_pop()
    if forward_only:
        pass
    else:
        nvtx.range_push("backward")
        scaler = torch.amp.grad_scaler.GradScaler()
        scaler.scale(loss + 1e-4 * z_loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    args = parse_args()
    if args.override:
        override = json.loads(args.override)
    else:
        override = {}
    cfg = apply_overrides(cfg, override)
    model = Transformer(**asdict(cfg.model), dtype=torch.float32)
    model = model.to(cfg.trainer.device)

    n_tries = 2
    warmup_steps = args.warmup_steps

    random_input = torch.randint(0, cfg.model.vocab_size, size=(1, cfg.data.context_length), dtype=torch.int32)
    random_input = random_input.to(cfg.trainer.device, non_blocking=True)

    for size, override in _SIZES.items():
        cfg = apply_overrides(cfg, override)
        model = Transformer(**asdict(cfg.model), dtype=torch.float32)
        model = model.to(cfg.trainer.device)
        # model.compile()
        optimizers = init_optimizers(model, cfg.model.d_model)

        seqs = [128, 256, 512, 1024]
        if size == "2.7b":
            seqs = [128, 256]
        for seq in seqs:
            random_input = torch.randint(0, cfg.model.vocab_size, size=(1, seq), dtype=torch.int32)
            random_input = random_input.to(cfg.trainer.device, non_blocking=True)
            runs = timeit.repeat(
                setup="""nvtx.range_push('warmup')
for _ in range(warmup_steps): 
    step(model, random_input, optimizers, forward_only)
nvtx.range_pop()""",
                stmt="""nvtx.range_push('benchmark')
step(model, random_input, optimizers, forward_only)
nvtx.range_pop()""",
                repeat=n_tries,
                number=1,
                globals={
                    "warmup_steps": warmup_steps,
                    "step": step,
                    "model": model,
                    "random_input": random_input,
                    "optimizers": optimizers,
                    "forward_only": args.forward_only,
                    "nvtx": nvtx,
                },
            )
            logger.info(f"{size=}, {seq=}: {np.mean(runs)=:.10f},{np.std(runs)=:.10f}")
