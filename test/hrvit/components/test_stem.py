import pytest
import tensorflow as tf

from hrvit.components.stem import Stem


@pytest.mark.parametrize('input_shape, halving_factor, output_channels, expected_output_shape', [
    ((1, 224, 224, 3), 0, 32, (1, 224, 224, 32)),
    ((1, 224, 224, 3), 1, 32, (1, 112, 112, 32)),
    ((1, 224, 224, 3), 2, 32, (1, 56, 56, 32)),
    ((1, 224, 224, 3), 3, 32, (1, 28, 28, 32)),
    ((1, 224, 224, 3), 4, 32, (1, 14, 14, 32)),
    ((1, 224, 224, 3), 5, 32, (1, 7, 7, 32)),
    ((1, 512, 512, 1), 0, 64, (1, 512, 512, 64)),
    ((1, 512, 512, 1), 1, 64, (1, 256, 256, 64)),
    ((1, 512, 512, 1), 2, 64, (1, 128, 128, 64)),
    ((1, 512, 512, 1), 3, 64, (1, 64, 64, 64)),
    ((1, 512, 512, 1), 4, 64, (1, 32, 32, 64)),
    ((1, 512, 512, 1), 5, 64, (1, 16, 16, 64)),
    ((1, 512, 512, 1), 6, 64, (1, 8, 8, 64)),
    ((1, 512, 512, 1), 7, 512, (1, 4, 4, 512)),
    ((1, 512, 512, 1), 8, 512, (1, 2, 2, 512)),
    ((1, 512, 512, 1), 9, 512, (1, 1, 1, 512)),
])
def test_stem_produces_expected_shape(
    input_shape: tuple[int, int, int, int],
    halving_factor: int,
    output_channels: int,
    expected_output_shape: tuple[int, int, int, int],
):
    stem = Stem(
        halving_factor=halving_factor,
        output_channels=output_channels,
    )
    x = tf.random.uniform(input_shape)
    y = stem(x)
    assert y.shape == expected_output_shape
