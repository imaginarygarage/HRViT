from keras import Model, layers

from .components.branch import BranchDefinition
from .components.stem import Stem
from .components.module import Module


def HRViT_b1():
    """
    HRViT-b1 model, 224x224 input resolution.
    """

    module_definitions = (
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
        ),
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=28, branch_width=28, branch_depth=64, transformer_count=1, attention_window_size=2, attention_head_count=1, mix_cfn_expansion_ratio=4),
        ),
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=28, branch_width=28, branch_depth=64, transformer_count=1, attention_window_size=2, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=14, branch_width=14, branch_depth=128, transformer_count=6, attention_window_size=7, attention_head_count=2, mix_cfn_expansion_ratio=4),
        ),
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=28, branch_width=28, branch_depth=64, transformer_count=1, attention_window_size=2, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=14, branch_width=14, branch_depth=128, transformer_count=6, attention_window_size=7, attention_head_count=2, mix_cfn_expansion_ratio=4),
        ),
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=28, branch_width=28, branch_depth=64, transformer_count=1, attention_window_size=2, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=14, branch_width=14, branch_depth=128, transformer_count=6, attention_window_size=7, attention_head_count=2, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=7, branch_width=7, branch_depth=256, transformer_count=2, attention_window_size=7, attention_head_count=4, mix_cfn_expansion_ratio=4),
        ),
        (
            BranchDefinition(branch_height=56, branch_width=56, branch_depth=32, transformer_count=1, attention_window_size=1, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=28, branch_width=28, branch_depth=64, transformer_count=1, attention_window_size=2, attention_head_count=1, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=14, branch_width=14, branch_depth=128, transformer_count=2, attention_window_size=7, attention_head_count=2, mix_cfn_expansion_ratio=4),
            BranchDefinition(branch_height=7, branch_width=7, branch_depth=256, transformer_count=2, attention_window_size=7, attention_head_count=4, mix_cfn_expansion_ratio=4),
        )
    )

    input = layers.Input((224, 224, 3))
    x = Stem(output_channels=32, halving_factor=2)(input)
    for branch_definitions in module_definitions:
        x = Module(branch_definitions=branch_definitions)(x)
    model = Model(inputs=input, outputs=x)

    return model
