import torch
from torch import nn

from llm.config import GPTConfig
from llm.layers.activations import GELU
from llm.layers.multi_head_attention import MultiHeadAttention
from llm.layers.norm import LayerNorm


class TransformerLayer(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.layer_norm_1 = LayerNorm(emb_dim=config.emb_dim)
        self.mh_attention = MultiHeadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            qkv_bias=config.qkv_bias,
            context_length=config.context_length,
            n_head=config.n_heads,
            dropout_rate=config.drop_rate,
        )
        self.dropout = nn.Dropout(config.drop_rate)
        self.layer_norm_2 = LayerNorm(emb_dim=config.emb_dim)
        self.ff = FeedForward(emb_dim=config.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm_1(x)
        x_context = self.mh_attention(x_norm)
        x_context = self.dropout(x_context)
        x_context = x_context + x

        x_norm_2 = self.layer_norm_2(x_context)
        x_ff = self.ff(x_norm_2)
        x_ff = self.dropout(x_ff)

        output = x_ff + x_context

        return output


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
