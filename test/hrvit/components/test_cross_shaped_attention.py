import pytest
import tensorflow as tf

from hrvit.components.cross_shaped_attention import CrossShapedAttention


@pytest.mark.parametrize('input_shape, window_size, head_count, expected_output_shape', [
    ((1, 16, 16, 8), 4, 1, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 4, 2, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 4, 4, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 8, 4, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 15, 4, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 16, 4, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 17, 4, (1, 16, 16, 8)),
    ((3, 32, 32, 10), 1, 5, (3, 32, 32, 10)),
])
def test_cross_shaped_attention_produces_expected_shape(
    input_shape: tuple[int, int, int, int],
    window_size: int,
    head_count: int,
    expected_output_shape: tuple[int, int, int, int],
):
    cross_shaped_attention = CrossShapedAttention(
        window_size=window_size,
        head_count=head_count,
    )
    token_block = tf.random.uniform(input_shape)

    context_block = cross_shaped_attention(token_block)

    assert context_block.shape == expected_output_shape
