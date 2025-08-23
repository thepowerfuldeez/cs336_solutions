from cs336_basics.config_schema import Config, DataConfig, ModelConfig, OptimConfig, TrainerConfig
from pathlib import Path

cfg = Config(
    data=DataConfig(),
    model=ModelConfig(),
    optim=OptimConfig(),
    trainer=TrainerConfig(),
)
