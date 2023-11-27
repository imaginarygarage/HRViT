import pytest
import tensorflow as tf

from hrvit.components.mix_cfn import MixCFN


@pytest.mark.parametrize('input_shape, expansion_ratio, expected_output_shape', [
    ((1, 16, 16, 8), 1, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 2, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 3, (1, 16, 16, 8)),
    ((1, 16, 16, 8), 4, (1, 16, 16, 8)),
    ((3, 16, 16, 8), 4, (3, 16, 16, 8)),
    ((1, 32, 32, 8), 4, (1, 32, 32, 8)),
    ((1, 16, 16, 4), 4, (1, 16, 16, 4)),
])
def test_transformer_produces_expected_shape(
    input_shape: float,
    expansion_ratio: int,
    expected_output_shape: float,
):
    mix_cfn = MixCFN(expansion_ratio=expansion_ratio)
    input = tf.random.uniform(input_shape)

    output = mix_cfn(input)

    assert output.shape == expected_output_shape
