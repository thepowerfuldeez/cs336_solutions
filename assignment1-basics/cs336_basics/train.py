import wandb

from cs336_basics.config_schema import Config
from cs336_basics.utils.config_tools import dataclass_to_nested_dict, wandb_run_name
from cs336_basics.training.trainer import Trainer
from cs336_basics.utils.logger import logger


def train(cfg: Config):
    run_name: str = wandb_run_name(cfg)
    logger.info(f"Training run: {run_name}")

    cfg.optim.cosine_steps = cfg.trainer.max_steps
    run = wandb.init(project=cfg.project, name=run_name, config=dataclass_to_nested_dict(cfg))
    trainer = Trainer(cfg, wandb=run)
    trainer.train()

    run.finish()


def test(cfg: Config):
    import torch
    trainer = Trainer(cfg)
    print(trainer.generate(torch.tensor([0, 1, 2]), 5, top_p=0.8, temperature=0.1))


from cs336_basics.configs.gpt_small import cfg
train(cfg)
# test(cfg)
