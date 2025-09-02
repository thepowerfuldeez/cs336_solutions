sudo apt update
sudo apt install zstd wget python3.12-dev -y
mkdir -p data_tokenized
wget https://pub-6fb04c2fe89645edb5eb2f9087d1d27b.r2.dev/owt_valid.npy -P data_tokenized
wget https://pub-6fb04c2fe89645edb5eb2f9087d1d27b.r2.dev/owt_train.npy.zstd -P data_tokenized
zstd -d data_tokenized/owt_train.npy.zstd -o data_tokenized/owt_train.npy
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
export UV_TORCH_BACKEND=auto
uv sync && uv tool install maturin && cd cs336_basics/fastsplit && maturin develop --release && cd ../../
uv run wandb login
uv run cs336_basics/train.py --override '{"model.vocab_size": 32000, "trainer.max_steps": 28300, "trainer.val_every": 5000, "trainer.save_every": 4000, "trainer.gradient_accumulation_steps": 2, "optim.use_muon": true, "optim.betas": [0.95, 0.99], "optim.muon_wd": 1.15e-4, "model.attn_qknorm": true, "model.layernorm_scaling": true, "data.context_length": 512, "data.batch_size": 96, "data.val_batch_size": 288, "optim.lr": 1.725e-2, "optim.lr_min": 1.15e-3, "model.n_layers": 8, "model.d_model": 768, "model.d_ff": 2048, "model.n_heads": 12, "model.attn_value_residual": true, "trainer.dtype": "bfloat16", "trainer.device": "cuda"}' --train-path data_tokenized/owt_train.npy --validation-path data_tokenized/owt_valid.npy