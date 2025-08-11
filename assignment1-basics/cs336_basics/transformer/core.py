import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from einops import einsum, rearrange


from cs336_basics.utils.logger import logger


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        in_features: final dim of the input
        out_features: final dim of the output
        """
        super().__init__()
        weight: Tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight: Float[Tensor, "out_features in_features"] = nn.Parameter(weight)
        std: float = 2.0 / (in_features + out_features)
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "batch ... in_features"]) -> Float[Tensor, "batch ... out_features"]:
        # x is row-wise vector
        out: Float[Tensor, "batch ... out_features"] = einsum(
            x, self.weight, "batch ... in_features, out_features in_features -> batch ... out_features"
        )
        return out


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        weight: Tensor = torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        self.weight: Float[Tensor, "vocab_size d_model"] = nn.Parameter(weight)
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        return torch.index_select(self.weight, dim=0, index=x.reshape(-1)).view(*x.size(), -1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        x is an activation from residual stream
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms: Float[Tensor, "... 1"] = torch.sqrt((x * x).mean(-1) + self.eps).unsqueeze(-1)
        out: Tensor = x / rms * self.gain
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # we would use larger matrix and split it by 2 later
        # round to 64 to use hardware better
        # d_ff: int = int(d_model * d_mult // 64 * 64)
        d_ff: int = int(d_ff // 64 * 64)
        self.up = Linear(d_model, d_ff * 2, device, dtype)
        self.down = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        projected: Float[Tensor, "... 2*d_ff"] = self.up(x)
        left, right = projected.chunk(2, dim=-1)
        return self.down(left * torch.sigmoid(left) * right)


def softmax(x: Tensor, dim: int = 0, temperature: float = 1.0) -> Tensor:
    o: Tensor = x - x.max(dim=dim, keepdim=True)[0]
    assert temperature > 0, "temperature must be more than 0"
    if temperature != 1.0:
        o /= temperature
    return o.exp() / o.exp().sum(dim=dim, keepdim=True)
