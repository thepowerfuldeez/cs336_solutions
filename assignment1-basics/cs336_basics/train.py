import argparse
import json

import wandb

from cs336_basics.config_schema import Config
from cs336_basics.utils.config_tools import dataclass_to_nested_dict, wandb_run_name
from cs336_basics.training.trainer import Trainer
from cs336_basics.utils.logger import logger
from cs336_basics.utils.config_tools import apply_overrides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--override", type=str, help='{"k": "v"} override to the cfg as dict')
    return p.parse_args()



def train(cfg: Config, args):
    override = json.loads(args.override)
    s = ""
    for k, v in override.items():
        s += f"{k.split('.')[-1]}={v}"
    run_name: str = wandb_run_name(cfg)
    run_name = f"{run_name}_{s}"
    logger.info(f"Training run: {run_name}")

    cfg = apply_overrides(cfg, override)
    overrides = {"optim.cosine_steps": cfg.trainer.max_steps}
    cfg = apply_overrides(cfg, overrides)
    cfg.trainer.save_dir = cfg.trainer.save_dir / s

    run = wandb.init(project=cfg.project, name=run_name, config=dataclass_to_nested_dict(cfg))
    trainer = Trainer(cfg, wandb=run)
    trainer.train()

    run.finish()


def test(cfg: Config):
    import torch
    trainer = Trainer(cfg)
    print(trainer.generate(torch.tensor([0, 1, 2]), 5, top_p=0.8, temperature=0.1))


from cs336_basics.configs.gpt_small import cfg
train(cfg, parse_args())
# test(cfg)
