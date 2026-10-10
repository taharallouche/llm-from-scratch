from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim {self.emb_dim} cannot be divided into {self.n_heads} heads."
            )


GPT_CONFIG_124M = GPTConfig()
