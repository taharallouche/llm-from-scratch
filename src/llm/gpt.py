import torch
from torch import nn

from llm.config import GPTConfig
from llm.layers.norm import LayerNorm
from llm.layers.transformer import TransformerLayer


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config

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
        _, seq_length = token_ids.shape

        x: torch.Tensor = self.tok_emb(token_ids)  # n_batch x seq_length x emb_dim
        positional_embs = self.pos_emb(torch.arange(seq_length))  # seq_length x emb_dim

        x = x + positional_embs
        x = self.drop_emb(x)

        x = self.trf_blocks(x)  # n_batch x seq_length x emb_dim
        x = self.final_norm(x)
        logits = self.out_head(x)  # n_batch x seq_length x vocab_size

        return logits
