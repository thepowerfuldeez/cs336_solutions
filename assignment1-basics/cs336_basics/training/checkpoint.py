from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from cs336_basics.config_schema import Config
from cs336_basics.utils.config_tools import save_config


def save_checkpoint(fpath: Path, cfg: Config | None, model: nn.Module, optimizer: Optimizer, iteration: int):
    torch.save(
        {
            "config": save_config(cfg) if cfg is not None else None,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        fpath,
    )


def load_checkpoint(fpath: Path, model: nn.Module, optimizer: Optimizer) -> int:
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
