import pytest
import tensorflow as tf

from hrvit.components.diversity_enhanced_shortcut import DiversityEnhancedShortcut


@pytest.mark.parametrize('input_shape, expected_output_shape', [
    ((1, 16, 16, 8), (1, 16, 16, 8)),
    ((1, 32, 32, 4), (1, 32, 32, 4)),
    ((3, 16, 16, 8), (3, 16, 16, 8)),
])
def test_transformer_produces_expected_shape(
    input_shape: float,
    expected_output_shape: float,
):
    des = DiversityEnhancedShortcut()
    input = tf.random.uniform(input_shape)

    output = des(input)

    assert output.shape == expected_output_shape
