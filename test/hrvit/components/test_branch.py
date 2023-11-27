import pytest
import tensorflow as tf

from hrvit.components.branch import Branch


@pytest.mark.parametrize('input_shapes, branch_dimensions, transformer_count, window_size, head_count, mix_cfn_expansion_ratio, expected_output_shape', [
    (((1, 16, 16, 32), (1, 32, 32, 16), (1, 64, 64, 8)), (32, 32, 8), 3, 2, 2, 4, (1, 32, 32, 8)),
])
def test_branch_produces_expected_shape(
    input_shapes: tuple[tuple[int, int, int, int], ...],
    branch_dimensions: tuple[int, int, int],
    transformer_count: int,
    window_size: int,
    head_count: int,
    mix_cfn_expansion_ratio: int,
    expected_output_shape: tuple[int, int, int, int],
):
    branch = Branch(
        branch_height=branch_dimensions[0],
        branch_width=branch_dimensions[1],
        branch_depth=branch_dimensions[2],
        transformer_count=transformer_count,
        attention_window_size=window_size,
        attention_head_count=head_count,
        mix_cfn_expansion_ratio=mix_cfn_expansion_ratio,
    )
    x = [tf.random.uniform(input_shape) for input_shape in input_shapes]
    y = branch(x)
    assert y.shape == expected_output_shape
