import pytest
import tensorflow as tf

from hrvit.components.transformer import Transformer


@pytest.mark.parametrize('input_shape, window_size, attention_heads, expected_output_shape', [
    ((1, 16, 16, 8), 4, 2, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 8, 4, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 15, 2, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 16, 2, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 17, 2, (1, 16, 16, 8)),
    ((3, 16, 16, 8), 4, 2, (3, 16, 16, 8)),
])
def test_transformer_produces_expected_shape(
    input_shape: float,
    window_size: int,
    attention_heads: int,
    expected_output_shape: float,
):
    transformer = Transformer(attention_window_size=window_size, attention_head_count=attention_heads)
    input = tf.random.uniform(input_shape)

    output = transformer(input)

    assert output.shape == expected_output_shape
