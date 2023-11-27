import pytest
import tensorflow as tf

from hrvit.components.cross_resolution_fusion import CrossResolutionFusion


@pytest.mark.parametrize('input_shapes, expected_output_shape', [
    (((1, 16, 16, 32), (1, 32, 32, 16), (1, 64, 64, 8)), (1, 32, 32, 8)),
    (((1, 8, 8, 32), (1, 48, 48, 16), (1, 64, 64, 2)), (1, 16, 16, 4)),
])
def test_cross_resolution_fusion_produces_expected_shape(
    input_shapes: tuple[tuple[int, int, int, int], ...],
    expected_output_shape: tuple[int, int, int, int],
):
    cross_resolution_fusion = CrossResolutionFusion(
        output_height=expected_output_shape[1],
        output_width=expected_output_shape[2],
        output_depth=expected_output_shape[3],
    )
    x = [tf.random.uniform(input_shape) for input_shape in input_shapes]

    y = cross_resolution_fusion(x)

    assert y.shape == expected_output_shape
