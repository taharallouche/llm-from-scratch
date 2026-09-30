import torch
from torch import nn

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ],
)

d_in = 3
d_out = 4

# Formula 1 / sqrt(...) x (Q x K.T) x V
# V: L x d_out
# K: L x d_out
# Q: L x d_out

# Q = X x W_Q + B_Q => W_Q: d_in x d_out

Q = nn.Linear(d_in, d_out)
K = nn.Linear(d_in, d_out)
V = nn.Linear(d_in, d_out)

attention_scores = Q(inputs) @ K(inputs).transpose(0, 1)

attention_weights = torch.softmax(attention_scores / d_out**0.5, dim=-1)


context = attention_weights @ V(inputs)


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q: torch.Tensor = self.W_query(x)
        K: torch.Tensor = self.W_key(x)
        V: torch.Tensor = self.W_value(x)

        d_k = K.shape[-1]

        attention_scores = Q @ K.transpose(-2, -1)
        attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=1)

        return attention_weights @ V


print(SelfAttention(d_in=3, d_out=2)(inputs))
