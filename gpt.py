from dataclasses import dataclass

import torch
from torch import nn

from multi_head_attention import MultiHeadAttention


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False


GPT_CONFIG_124M = GPTConfig()


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.context_size = config.context_length
        self.tok_emb = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.emb_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=config.context_length, embedding_dim=config.emb_dim
        )
        self.drop_emb = nn.Dropout(p=config.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerLayer(config=config) for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNorm(emb_dim=config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        n_batch, seq_length = token_ids.shape

        x: torch.Tensor = self.tok_emb(token_ids)  # n_batch x seq_length x emb_dim
        positional_embs = self.pos_emb(torch.arange(seq_length))  # seq_length x emb_dim

        x = x + positional_embs
        x = self.drop_emb(x)

        x = self.trf_blocks(x)  # n_batch x seq_length x emb_dim
        x = self.final_norm(x)
        logits = self.out_head(x)  # n_batch x seq_length x vocab_size

        return logits


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


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


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


### Personal attempt to implement Transformer Layer based on Figure 4.13


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


def generate_text_simple(
    model: GPTModel, token_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        input_ = token_ids[:, -model.context_size :]  # n_batch, context_length
        logits = model(input_)  # n_batch, context_size, vocab_size
        logits = logits[:, -1, :]  # n_batch, vocab_size (only logits of last token)
        probabilities = torch.softmax(logits, dim=-1)  # n_batch, vocab_size
        next_token_ids = torch.argmax(probabilities, dim=-1, keepdim=True)  # n_batch, 1
        token_ids = torch.concat(
            [token_ids, next_token_ids], dim=1
        )  # n_batch, context_length + 1
    return token_ids


if __name__ == "__main__":
    # x = torch.randn(5, GPT_CONFIG_124M.context_length, GPT_CONFIG_124M.emb_dim)
    #
    # tf = TransformerLayer(config=GPT_CONFIG_124M)
    #
    # output: torch.Tensor = tf(x)
    #
    # print(output.shape)

    gpt = GPTModel(config=GPT_CONFIG_124M)

    batch = torch.tensor(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]
    )  # n_batch:2, num_tokens = 5 < context_length

    with_prediction = generate_text_simple(gpt, batch, 2)

    print(with_prediction)
