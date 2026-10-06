import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=True, keepdim=True)

        normalized = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * normalized + self.shift
