import pytest
from tokenizers import Tokenizer

from llm.fine_tuning.classification.classifier import setup_binary_classification_model
from llm.gpt import GPTModel


def test_classify_can_handle_one_token_input(
    small_model: GPTModel, tokenizer: Tokenizer
) -> None:
    # Given
    from llm.fine_tuning.classification.inference import classify

    model = setup_binary_classification_model(model=small_model)
    text = "o"
    classes = [0, 1]

    # When
    output = classify(model=model, tokenizer=tokenizer, text=text, classes=classes)

    # Then
    assert output in classes


@pytest.mark.parametrize(["threshold", "expected_class"], [(0.0, 1), (1.0, 0)])
def test_classify_respects_threshold(
    threshold: float, expected_class: int, small_model: GPTModel, tokenizer: Tokenizer
) -> None:
    # Given
    from llm.fine_tuning.classification.inference import classify

    model = setup_binary_classification_model(model=small_model)
    text = "text"
    classes = [0, 1]

    # When
    output = classify(
        model=model,
        tokenizer=tokenizer,
        text=text,
        threshold=threshold,
        classes=classes,
    )

    # Then
    assert expected_class == output
