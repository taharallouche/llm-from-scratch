from collections.abc import Sequence
from functools import partial

import tiktoken
import torch

from llm.config import GPTConfig
from llm.fine_tuning.classification.classifier import setup_binary_classification_model
from llm.gpt import GPTModel
from llm.io.utils import checkpoint_path
from llm.training.pipeline import text_to_token_ids


def classify(
    model: GPTModel,
    tokenizer: tiktoken.Encoding,
    text: str,
    classes: Sequence[str],
    threshold: float = 0.5,
) -> str:
    token_ids = text_to_token_ids(text, tokenizer=tokenizer)
    model.eval()
    with torch.no_grad():
        output: torch.Tensor = model(token_ids)
    last_logits = output.squeeze()[-1, :]
    probas = torch.softmax(last_logits, dim=0)
    return classes[int(probas[1] > threshold)]


if __name__ == "__main__":
    model = setup_binary_classification_model(
        model=GPTModel(GPTConfig(drop_rate=0.0, qkv_bias=True)),
    )

    state_dict = torch.load(checkpoint_path("movie_classifier_left_padded.pth"))
    model.load_state_dict(state_dict)

    tokenizer = tiktoken.get_encoding("gpt2")

    classify_ = partial(
        classify,
        model=model,
        tokenizer=tokenizer,
        classes=["negative", "positive"],
        threshold=0.5,
    )

    text = "Sweet"
    print(f"{text}: {classify_(text=text)}")
