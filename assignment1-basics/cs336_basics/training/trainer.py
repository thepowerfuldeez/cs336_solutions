import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from jaxtyping import Int
from tqdm.auto import tqdm

from cs336_basics.transformer import Transformer
from cs336_basics.training import (
    load_checkpoint,
    save_checkpoint,
    MemoryMappedDataset,
    AdamW,
    get_cosine_lr,
    clip_grad_norm_,
    cross_entropy,
)
from cs336_basics.config_schema import Config
from cs336_basics.utils.logger import logger
from cs336_basics.utils.config_tools import load_config, save_config

torch.set_float32_matmul_precision("high")


def mem(tag):
    alloc = torch.cuda.memory_allocated() / 1e9
    resv = torch.cuda.memory_reserved() / 1e9
    logger.info(f"{tag}: allocated={alloc:.2f} GB, reserved={resv:.2f} GB")


class Trainer:
    """
    class that takes cfg: Config and runs training
    """

    def __init__(self, cfg: Config | None = None, load_from: str | None = None, wandb: Any | None = None):
        if cfg is None:
            assert load_from is not None, "you must load from checkpoint if cfg is None"
            self.cfg = load_config(torch.load(load_from)["config"])
        else:
            logger.info("Loading from config")
            self.cfg = cfg
        self.model = Transformer(
            self.cfg.model.n_layers,
            self.cfg.model.vocab_size,
            self.cfg.model.d_model,
            self.cfg.model.n_heads,
            self.cfg.model.d_ff,
            self.cfg.model.theta,
            torch.device(self.cfg.trainer.device),
            getattr(torch, self.cfg.trainer.dtype),
        )
        self.model.to(self.cfg.trainer.device)
        self.model.compile()
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.cfg.optim.lr, betas=self.cfg.optim.betas, weight_decay=self.cfg.optim.wd
        )
        logger.info(f"Model is created with hparams {self.cfg.model}")
        self.train_dataset = MemoryMappedDataset(
            self.cfg.data.train_path,
            self.cfg.data.context_length,
            torch.device(self.cfg.trainer.device),
            self.cfg.data.seed,
        )
        self.val_dataset = MemoryMappedDataset(
            self.cfg.data.validation_path,
            self.cfg.data.context_length,
            torch.device(self.cfg.trainer.device),
            self.cfg.data.seed,
        )
        self.save_dir = Path(self.cfg.trainer.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.iteration = 0
        self.wandb = wandb
        load_from = load_from or self.cfg.trainer.load_from
        if load_from is not None:
            self.load_state(load_from)

    @property
    def tokens_processed(self):
        return self.iteration * self.cfg.data.batch_size * self.cfg.data.context_length

    def load_state(self, path: Path):
        logger.info(f"Loading state from {str(path)}")
        self.iteration = load_checkpoint(path, self.model, self.optimizer)

    def save_state(self):
        logger.info(f"Saving training state at iter={self.iteration}")
        save_checkpoint(self.save_dir / f"{self.iteration}.pt", self.cfg, self.model, self.optimizer, self.iteration)

    def log(self, **data):
        for k, v in data.items():
            logger.info(f"{k}: {v}")
        if self.wandb is not None:
            self.wandb.log(data)

    def generate(
        self,
        prompt: Int[Tensor, "bs seq"],
        eos_token_id: int,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 32,
    ):
        """
        Perform decoding with nucleous sampling and temperature
        """
        return self.model.generate(prompt, eos_token_id, top_p=top_p, temperature=temperature, max_steps=max_steps)

    def validate(self):
        mem("before validation")
        self.model.eval()
        val_iters = 0
        ds_perplexity = torch.zeros((1,), device="cpu")
        with torch.inference_mode():
            for inputs, targets in tqdm(
                self.val_dataset.get_iterator(self.cfg.data.val_batch_size),
                total=len(self.val_dataset) // (self.cfg.data.val_batch_size * self.cfg.data.context_length),
                desc="Running validation",
            ):
                logits = self.model(inputs)
                perplexity = cross_entropy(logits, targets).exp().cpu()
                ds_perplexity += perplexity
                val_iters += 1
        mem("after validation")
        return ds_perplexity.item() / val_iters

    def train_step(self, inputs, targets):
        self.model.train()
        iter_lr = get_cosine_lr(
            self.iteration,
            self.cfg.optim.lr,
            self.cfg.optim.lr_min,
            self.cfg.optim.warmup_steps,
            self.cfg.optim.cosine_steps,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = iter_lr
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.trainer.max_grad_norm)
        _, update_ratio = self.optimizer.step()
        return {"train_loss": loss.item(), "grad_norm": grad_norm, "lr": iter_lr, "update_ratio": update_ratio}

    def train(self):
        logger.info("Starting training loop")
        while self.iteration < self.cfg.trainer.max_steps:
            if self.iteration % self.cfg.trainer.save_every == 0:
                self.save_state()
            if self.iteration > 0 and self.iteration % self.cfg.trainer.val_every == 0:
                perplexity = self.validate()
                self.log(val_perplexity=perplexity)

            inputs, targets = self.train_dataset.get_batch(self.cfg.data.batch_size)
            t0 = time.monotonic()
            step_stats = self.train_step(inputs, targets)
            t1 = time.monotonic()
            if self.iteration % self.cfg.trainer.log_every == 0:
                logger.info(f"Train iteration {self.iteration}")
                self.log(
                    **{
                        "train/loss": step_stats["train_loss"],
                        "train/grad_norm": step_stats["grad_norm"],
                        "train/learning_rate": step_stats["lr"],
                        "train/update_ratio": step_stats["update_ratio"],
                        "train/step": self.iteration,
                        "train/step_time": t1 - t0,
                        "train/tokens_processed": self.tokens_processed,
                    }
                )
            self.iteration += 1
        self.save_state()
        self.validate()
