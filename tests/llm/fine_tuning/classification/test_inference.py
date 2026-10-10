import tiktoken

from llm.fine_tuning.classification.classifier import setup_binary_classification_model
from llm.gpt import GPTModel


def test_classify_can_handle_one_token_input(
    small_model: GPTModel, tokenizer: tiktoken.Encoding
) -> None:
    # Given
    from llm.fine_tuning.classification.inference import classify

    model = setup_binary_classification_model(model=small_model)
    text = "o"
    classes = [0, 1]

    # When
    output = classify(model=model, tokenizer=tokenizer, text=text)

    # Then
    assert output in classes
