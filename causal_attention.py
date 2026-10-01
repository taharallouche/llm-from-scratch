import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout_rate: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_token, _ = x.shape  # batch size, num_tokens, dim_embed

        Q: torch.Tensor = self.W_query(x)
        K: torch.Tensor = self.W_key(x)
        V: torch.Tensor = self.W_value(x)

        d_k = K.shape[-1]

        attention_scores = Q @ K.transpose(-2, -1)

        attention_scores.masked_fill_(
            self.mask.bool()[:num_token, :num_token], -torch.inf
        )
        attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=1)

        return self.dropout(attention_weights) @ V


if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [
                [0.43, 0.15, 0.89],  # Your
                [0.55, 0.87, 0.66],  # journey
                [0.57, 0.85, 0.64],  # starts
                [0.22, 0.58, 0.33],  # with
                [0.77, 0.25, 0.10],  # one
                [0.05, 0.80, 0.55],  # step
            ],
            [
                [0.43, 0.15, 0.89],  # Your
                [0.55, 0.87, 0.66],  # journey
                [0.57, 0.85, 0.64],  # starts
                [0.22, 0.58, 0.33],  # with
                [0.77, 0.25, 0.10],  # one
                [0.05, 0.80, 0.55],  # step
            ],
        ],
    )
    print(
        CausalAttention(
            d_in=3, d_out=2, context_length=inputs.shape[1], dropout_rate=0.4
        )(inputs)
    )
