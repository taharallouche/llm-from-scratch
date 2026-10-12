from collections.abc import Sequence
from functools import partial

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer

from llm.config import GPTConfig
from llm.fine_tuning.classification.classifier import setup_binary_classification_model
from llm.gpt import GPTModel
from llm.io.utils import experiments_path
from llm.training.pipeline import text_to_token_ids


def classify[Class_](
    model: GPTModel,
    tokenizer: Tokenizer,
    text: str,
    classes: Sequence[Class_] = (0, 1),  # type: ignore[assignment]
    threshold: float = 0.5,
) -> Class_:
    token_ids = text_to_token_ids(text, tokenizer=tokenizer)
    model.eval()
    with torch.no_grad():
        output: torch.Tensor = model(token_ids)
    last_logits = output[0, -1, :]
    probas = torch.softmax(last_logits, dim=0)
    return classes[int(probas[1] > threshold)]


if __name__ == "__main__":
    model = setup_binary_classification_model(
        model=GPTModel(GPTConfig(drop_rate=0.0, qkv_bias=True)),
    )

    load_model(
        model, experiments_path() / "movie_reviews" / "movie_reviews.safetensors"
    )

    tokenizer = Tokenizer.from_pretrained("gpt2")

    classify_ = partial(
        classify,
        model=model,
        tokenizer=tokenizer,
        classes=["negative", "positive"],
        threshold=0.5,
    )

    text = "Bad scenario"
    print(f"{text}: {classify_(text=text)}")
