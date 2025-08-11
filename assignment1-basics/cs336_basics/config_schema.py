from dataclasses import dataclass
from typing import Literal
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    train_path: str | Path
    validation_path: str
    batch_size: int = 1
    context_length: int = 1024
    seed: int = 42


@dataclass(frozen=False)
class OptimConfig:
    lr: float = 3e-4
    wd: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.95)
    lr_min: float = 1e-6
    warmup_steps: int = 100
    cosine_steps: int = 10_000


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 1024
    d_ff: int = 4096
    n_layers: int = 24
    n_heads: int = 16
    vocab_size: int = 10_000
    theta: float = 10_000


@dataclass(frozen=True)
class TrainerConfig:
    load_from: str | None = None
    device: str = "mps"
    dtype: Literal["float32", "bfloat16"] = "float32"
    max_steps: int = 200_000
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    # run_name: str = "{date}_{optim.lr}"  # template
    run_name: str = "{date}"  # template
    save_dir: str | Path = Path(__file__).parent / "checkpoints"
    # save every n steps
    save_every: int = 100
    # validate every n steps
    val_every: int = 100
    # log train metrics every n steps
    log_every: int = 10


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    trainer: TrainerConfig
    project: str = "cs336"  # for wandb
