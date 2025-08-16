from pathlib import Path
from argparse import ArgumentParser

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training.trainer import Trainer


def parse_args():
    p = ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="/home/george/cs336_solutions/assignment1-basics/cs336_basics/checkpoints/betas=[0.9, 0.99]/8000.pt",
    )
    p.add_argument("--tokenizer", default="/home/george/cs336_solutions/assignment1-basics/tokenizer/tinystories")
    p.add_argument("--top-p", default=0.95, type=float)
    p.add_argument("--temperature", default=0.0, type=float)
    return p.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(load_from=args.checkpoint)
    tokenizer = Tokenizer.from_files(
        Path(args.tokenizer) / "vocab.pickle", Path(args.tokenizer) / "merges.pickle", special_tokens=["<|endoftext|>"]
    )
    eos_token_id = tokenizer.encode(tokenizer.special_tokens[0])[0]

    print("EOS", eos_token_id)

    prompt = torch.tensor(tokenizer.encode("Once")).unsqueeze(0).to(trainer.cfg.trainer.device)
    generated = trainer.generate(
        prompt,
        eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        max_steps=512
    )
    print(generated)

    print(tokenizer.decode(generated[0].cpu().tolist()))


if __name__ == "__main__":
    main()
