from cs336_basics.config_schema import Config, ModelConfig, DataConfig, TrainerConfig, OptimConfig
from cs336_basics.configs.base import cfg as base

cfg = Config(
    data=DataConfig(
        base.data.train_path,
        base.data.validation_path,
        batch_size=160,
        val_batch_size=384,
        context_length=256,
        seed=42,
    ),
    model=ModelConfig(d_model=512, d_ff=1344, n_layers=4, n_heads=16),
    optim=OptimConfig(lr=7e-3),
    trainer=TrainerConfig(log_every=50, save_every=1000, val_every=1000, max_steps=8000),
    project=base.project,
)
