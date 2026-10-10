import pytest


def test_GPTConfig_validates_emb_dim_and_n_heads_consistency() -> None:
    # Given
    from llm.config import GPTConfig

    # When / Then
    with pytest.raises(ValueError, match="cannot be divided"):
        GPTConfig(emb_dim=10, n_heads=9)
