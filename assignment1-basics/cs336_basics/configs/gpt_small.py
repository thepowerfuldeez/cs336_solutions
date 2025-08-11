from cs336_basics.config_schema import Config, ModelConfig
from cs336_basics.configs.base import cfg as base

cfg = Config(
    data=base.data,
    model=ModelConfig(d_model=768, n_layers=12, n_heads=12),
    optim=base.optim,
    trainer=base.trainer,
    project=base.project,
)
