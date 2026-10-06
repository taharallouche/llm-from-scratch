import pytest
import torch


@pytest.mark.parametrize("n_head", [1, 2])
def test_MultiHeadAttention_forward_output_shape(n_head: int) -> None:
    # Given
    from llm.layers.multi_head_attention import MultiHeadAttention

    inputs = torch.tensor(
        [
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
                [0.77, 0.25, 0.10],
                [0.05, 0.80, 0.55],
            ],
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
                [0.77, 0.25, 0.10],
                [0.05, 0.80, 0.55],
            ],
        ],
    )

    mha = MultiHeadAttention(
        d_in=3,
        d_out=4,
        qkv_bias=False,
        context_length=6,
        n_head=n_head,
        dropout_rate=0.1,
    )
    mha.eval()

    expected_shape = (2, 6, 4)

    # When
    with torch.no_grad():
        output: torch.Tensor = mha(inputs)

    # Then
    assert expected_shape == output.shape
