from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(fpath: Path, model: nn.Module, optimizer: Optimizer, iteration: int):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, fpath)


def load_checkpoint(fpath: Path, model: nn.Module, optimizer: Optimizer) -> int:
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
