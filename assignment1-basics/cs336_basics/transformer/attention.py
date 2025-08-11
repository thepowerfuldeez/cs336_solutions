import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from einops import einsum, rearrange

from cs336_basics.transformer.core import Linear, softmax
from cs336_basics.transformer.rope import RotatyPositionalEmbedding


def sdpa(
    q: Float[Tensor, "... seq_len d_k"],
    k: Float[Tensor, "... seq_len d_k"],
    v: Float[Tensor, "... seq_len d_k"],
    mask: Float[Tensor, "... seq_len seq_len"] | None = None,
) -> Float[Tensor, "... seq_len d_k"]:
    attn_scores = einsum(q, k, "... s1 d_k, ... s2 d_k -> ... s1 s2")
    attn_scores /= torch.sqrt(torch.tensor(q.size(-1)))
    if mask is not None:
        attn_scores.masked_fill_(~mask, float("-inf"))
    return softmax(attn_scores, -1) @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, theta: float = 10_000, max_seq_len: int = 4096, device=None, dtype=None
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Hidden dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.rope = RotatyPositionalEmbedding(theta, d_k=d_model // n_heads, max_seq_len=max_seq_len, device=device)
        self.qkv = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        self.out = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, "b seq d"], token_positions: Int[Tensor, "b seq"] | None = None
    ) -> Float[Tensor, "b seq d"]:
        Q, K, V = self.qkv(x).chunk(3, -1)
        seq_len = Q.size(1)
        Q = rearrange(Q, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads)
        K = rearrange(K, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        V = rearrange(V, "b seq (h head_d) -> (h b) seq head_d", h=self.n_heads)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=Q.dtype), diagonal=0).unsqueeze(0).bool()
        attn: Float[Tensor, "(h b) seq head_d"] = sdpa(Q, K, V, mask)
        return self.out(rearrange(attn, "(h b) seq head_d -> b seq (h head_d)", h=self.n_heads))


if __name__ == "__main__":
    sa = MultiHeadSelfAttention(1024, 8)
    x = torch.randn(4, 64, 1024)
    token_positions = torch.arange(0, 64).unsqueeze(0)
    sa(x, token_positions)
