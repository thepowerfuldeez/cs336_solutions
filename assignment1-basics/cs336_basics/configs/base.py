from cs336_basics.config_schema import Config, DataConfig, ModelConfig, OptimConfig, TrainerConfig
from pathlib import Path

cfg = Config(
    data=DataConfig(
        train_path=str(Path(__file__).parent.parent.parent / "data_tokenized/TinyStoriesV2-GPT4-train.npy"),
        validation_path=str(Path(__file__).parent.parent.parent / "data_tokenized/TinyStoriesV2-GPT4-valid.npy"),
        batch_size=256,
    ),
    model=ModelConfig(),
    optim=OptimConfig(),
    trainer=TrainerConfig(),
)
