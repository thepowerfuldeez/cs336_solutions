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


class Trainer:
    """
    class that takes cfg: Config and runs training
    """

    def __init__(self, cfg: Config, wandb: Any | None = None):
        self.cfg = cfg
        self.model = Transformer(
            cfg.model.n_layers,
            cfg.model.vocab_size,
            cfg.model.d_model,
            cfg.model.n_heads,
            cfg.model.d_ff,
            cfg.model.theta,
            torch.device(cfg.trainer.device),
            getattr(torch, cfg.trainer.dtype),
        )
        self.model.to(cfg.trainer.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.wd
        )
        logger.info(f"Model is created with hparams {cfg.model}")
        self.train_dataset = MemoryMappedDataset(
            cfg.data.train_path, cfg.data.context_length, torch.device(cfg.trainer.device), cfg.data.seed
        )
        self.val_dataset = MemoryMappedDataset(
            cfg.data.validation_path, cfg.data.context_length, torch.device(cfg.trainer.device), cfg.data.seed
        )
        self.save_dir = cfg.trainer.save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.iteration = 0
        self.wandb = wandb
        if cfg.trainer.load_from is not None:
            self.load_state(cfg.trainer.load_from)

    def load_state(self, path: Path):
        logger.info(f"Loading state from {str(path)}")
        self.iteration = load_checkpoint(path, self.model, self.optimizer)

    def save_state(self):
        logger.info(f"Saving training state at iter={self.iteration}")
        save_checkpoint(self.save_dir / f"{self.iteration}.pt", self.model, self.optimizer, self.iteration)

    def log(self, k: str, v: Any):
        logger.info(f"{k}: {v}")
        if self.wandb is not None:
            self.wandb.log({k: v})

    def generate(self, prompt: Int[Tensor, "bs seq"], eos_token_id: int, top_p: float = 1.0, temperature: float = 1.0):
        """
        Perform decoding with nucleous sampling and temperature
        """
        return self.model.generate(prompt, eos_token_id, top_p=top_p, temperature=temperature)

    def validate(self):
        val_iters = 0
        ds_perplexity = torch.zeros((1,), device=self.cfg.trainer.device)
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_dataset.get_iterator(self.cfg.data.batch_size),
                total=len(self.val_dataset) // self.cfg.data.batch_size,
                desc="Running validation",
            ):
                logits = self.model(inputs)
                perplexity = cross_entropy(logits, targets).exp()
                ds_perplexity += perplexity
                val_iters += 1
        return ds_perplexity.item() / val_iters

    def train_step(self, inputs, targets):
        iter_lr = get_cosine_lr(
            self.iteration,
            self.cfg.optim.lr,
            self.cfg.optim.lr_min,
            self.cfg.optim.warmup_steps,
            self.cfg.optim.cosine_steps,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = iter_lr
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.trainer.max_grad_norm)
        self.optimizer.step()
        return loss.item(), grad_norm, iter_lr

    def train(self):
        logger.info("Starting training loop")
        while self.iteration < self.cfg.trainer.max_steps:
            if self.iteration % self.cfg.trainer.save_every == 0:
                self.save_state()
            if self.iteration % self.cfg.trainer.val_every == 0:
                perplexity = self.validate()
                self.log("val_perplexity", perplexity)

            inputs, targets = self.train_dataset.get_batch(self.cfg.data.batch_size)
            loss_value, grad_norm, iter_lr = self.train_step(inputs, targets)
            if self.iteration % self.cfg.trainer.log_every == 0:
                logger.info(f"Train iteration {self.iteration}")
                self.log("train/loss", loss_value)
                self.log("grad_norm", grad_norm)
                self.log("learning_rate", iter_lr)
            self.iteration += 1
