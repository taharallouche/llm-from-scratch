import pytest
import tiktoken

from llm.config import GPTConfig
from llm.gpt import GPTModel


@pytest.fixture(scope="function")
def small_model_config() -> GPTConfig:
    return GPTConfig(context_length=10, emb_dim=6, n_heads=2, n_layers=3)


@pytest.fixture(scope="function")
def small_model(small_model_config: GPTConfig) -> GPTModel:
    return GPTModel(small_model_config)


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")
