import torch
from torch import nn

from llm.causal_attention import CausalAttention


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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        qkv_bias: bool,
        context_length: int,
        n_head: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.weight_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.weight_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.weight_V = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask", torch.triu(torch.ones((context_length, context_length)), diagonal=1)
        )

        self.d_out = d_out
        self.n_head = n_head
        self.d_head = d_out // n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, num_tokens, _ = x.shape

        Q: torch.Tensor = self.weight_Q(x)
        K: torch.Tensor = self.weight_K(x)
        V: torch.Tensor = self.weight_V(x)

        Q = Q.view(n_batch, num_tokens, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(n_batch, num_tokens, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(n_batch, num_tokens, self.n_head, self.d_head).transpose(1, 2)

        # attention_scores: n_batch x n_head x num_tokens x num_tokens
        attention_scores = Q @ K.transpose(-2, -1)
        masked_scores = attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(masked_scores / self.d_head**0.5, dim=-1)
        attention_weights: torch.Tensor = self.dropout(attention_weights)

        context = attention_weights @ V

        context = (
            context.transpose(1, 2).contiguous().view(n_batch, num_tokens, self.d_out)
        )

        return self.linear_proj(context)
