import pytest
import tensorflow as tf

from hrvit.components.branch import BranchDefinition
from hrvit.components.module import Module


@pytest.mark.parametrize('branch_definitions, expected_output_shapes', [
    (
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=28, branch_width=28, branch_depth=64, transformer_count=1, attention_window_size=2, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=14, branch_width=14, branch_depth=128, transformer_count=2, attention_window_size=7, attention_head_count=2, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=7, branch_width=7, branch_depth=256, transformer_count=2, attention_window_size=7, attention_head_count=4, mix_cfn_expansion_ratio=4),
        ), ((1, 56, 56, 32), (1, 28, 28, 64), (1, 14, 14, 128), (1, 7, 7, 256)),
    )
])
def test_branch_produces_expected_shape(
    branch_definitions: tuple[BranchDefinition, ...],
    expected_output_shapes: tuple[tuple[int, int, int, int], ...],
):
    module = Module(branch_definitions=branch_definitions)
    input = tf.random.uniform((1, 224, 224, 3))
    output = module(input)
    assert tuple([output.shape for output in output]) == expected_output_shapes
