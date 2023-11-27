import pytest
import tensorflow as tf

from hrvit.components.patch_embed import PatchEmbed


@pytest.mark.parametrize('input_shape, token_depth, expected_output_shape', [
    ((1, 64, 64, 16), 32, (1, 64, 64, 32)),
    ((1, 64, 64, 32), 32, (1, 64, 64, 32)),
    ((1, 64, 64, 64), 32, (1, 64, 64, 32)),
    ((1, 128, 128, 8), 16, (1, 128, 128, 16)),
    ((1, 128, 128, 16), 16, (1, 128, 128, 16)),
    ((1, 128, 128, 32), 16, (1, 128, 128, 16)),
])
def test_patch_embed_produces_expected_shape(
    input_shape: tuple[int, int, int, int],
    token_depth: int,
    expected_output_shape: tuple[int, int, int],
):
    patch_embed = PatchEmbed(token_depth)

    x = tf.random.uniform(input_shape)
    y = patch_embed(x)

    assert y.shape == expected_output_shape
