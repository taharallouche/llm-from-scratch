import torch
from torch import nn

from llm.educative.causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        qkv_bias: bool,
        context_length: int,
        dropout_rate: float,
        n_heads: int,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout_rate, qkv_bias)
                for _ in range(n_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([head(x) for head in self.heads], dim=-1)
