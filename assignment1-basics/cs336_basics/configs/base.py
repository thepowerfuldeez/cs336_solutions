from cs336_basics.config_schema import Config, DataConfig, ModelConfig, OptimConfig, TrainerConfig

cfg = Config(
    data=DataConfig(
        train_path="/Users/george/Projects/learning/cs336_solutions/assignment1-basics/data_tokenized/TinyStoriesV2-GPT4-train.npy",
        validation_path="/Users/george/Projects/learning/cs336_solutions/assignment1-basics/data_tokenized/TinyStoriesV2-GPT4-valid.npy",
        batch_size=1,
    ),
    model=ModelConfig(),
    optim=OptimConfig(),
    trainer=TrainerConfig(),
)
