"""Rotary Positional Embeddings"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum


def get_cos_sin(m: int, theta_base: float, d_k: int, device) -> Float[Tensor, "d_k d_k"]:
    """
    Make thetas matrix for rope

    m - position
    d_k - number of dimentions
    """
    # for i = 1 (second sequence)
    thetas = torch.tensor(theta_base).unsqueeze(0).repeat(d_k // 2)
    to_pow = torch.tensor([-2 * (i - 1) / d_k for i in range(1, d_k // 2 + 1)])
    thetas = m * thetas.pow(to_pow)

    matrix = torch.zeros(d_k, d_k)
    for k in range(1, d_k // 2 + 1):
        x = torch.cos(thetas[k - 1])
        y = torch.sin(thetas[k - 1])
        matrix[2 * k - 2 : 2 * k - 1, 2 * k - 2 : 2 * k] = torch.tensor([x, -y])
        matrix[2 * k - 1 : 2 * k, 2 * k - 2 : 2 * k] = torch.tensor([y, x])
    return matrix


class RotatyPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        full_rot_m = torch.zeros(max_seq_len, d_k, d_k, dtype=torch.float32, device=device)
        for i in range(max_seq_len):
            full_rot_m[i] = get_cos_sin(i, theta, d_k, device)
        self.register_buffer("full_rot_m", full_rot_m, persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_k"]:
        in_dtype = x.dtype
        if token_positions is not None:
            rot = self.full_rot_m[token_positions]
        else:
            rot = self.full_rot_m[: x.size(-2)]
        rotated: Float[Tensor, "... seq_len d_k"] = einsum(
            x.to(torch.float32), rot, "... seq_len d1, ... seq_len d2 d1 -> ... seq_len d2"
        )
        return rotated.to(in_dtype)


if __name__ == "__main__":
    rot = RotatyPositionalEmbedding(10, 32, 16)
