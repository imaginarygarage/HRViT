from dataclasses import dataclass
from keras import layers
from tensorflow import Tensor
from typing import Any, Optional

from .cross_resolution_fusion import CrossResolutionFusion
from .patch_embed import PatchEmbed
from .transformer import Transformer


@dataclass
class BranchDefinition:
    branch_height: int = 224
    branch_width: int = 224
    branch_depth: int = 32
    transformer_count: int = 1
    attention_window_size: int = 1
    attention_head_count: int = 2
    mix_cfn_expansion_ratio: int = 4


class Branch(layers.Layer):
    """
    A branch of the HRViT module.

    Args:
        branch_height: the height of the image in this branch.
        branch_width: the width of the image in this branch.
        branch_depth: the depth of the image in this branch (number of channels).
        transformer_count: number of transformer layers in this branch.
        attention_window_size: The independent dimensions of the vertical and
            horizontal attention windows.
        attention_head_count: Number of attention heads. Must divide
            branch_depth / 2 evenly.
        mix_cfn_expansion_ratio: expansion ratio of the MixCFN layers.
        name: Name of the layer.
    """

    def __init__(
        self,
        branch_height: int = 224,
        branch_width: int = 224,
        branch_depth: int = 32,
        transformer_count: int = 1,
        attention_window_size: int = 1,
        attention_head_count: int = 2,
        mix_cfn_expansion_ratio: int = 4,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.branch_height = branch_height
        self.branch_width = branch_width
        self.branch_depth = branch_depth
        self.transformer_count = transformer_count
        self.attention_window_size = attention_window_size
        self.attention_head_count = attention_head_count
        self.mix_cfn_expansion_ratio = mix_cfn_expansion_ratio
        self.cross_resolution_fusion = CrossResolutionFusion(
            output_height=branch_height,
            output_width=branch_width,
            output_depth=branch_depth,
        )
        self.patch_embedding = PatchEmbed(token_depth=branch_depth)
        self.transformers = [
            Transformer(
                attention_window_size=attention_window_size,
                attention_head_count=attention_head_count,
                mix_cfn_expansion_ratio=mix_cfn_expansion_ratio,
            ) for _ in range(transformer_count)
        ]
        self.layer_normalization = layers.LayerNormalization()

    def call(self, inputs: tuple[Tensor, ...]) -> Tensor:
        """
        Input shapes: N x (batch_size, height, width, token_depth)
        Output shape: (batch_size, height, width, token_depth)
        """
        x = self.cross_resolution_fusion(inputs)
        x = self.patch_embedding(x)
        for transformer in self.transformers:
            x = transformer(x)
        output = self.layer_normalization(x)
        return output

    @classmethod
    def from_definition(cls, definition: BranchDefinition, name: Optional[str] = None, **kwargs) -> 'Branch':
        return cls(
            branch_height=definition.branch_height,
            branch_width=definition.branch_width,
            branch_depth=definition.branch_depth,
            transformer_count=definition.transformer_count,
            attention_window_size=definition.attention_window_size,
            attention_head_count=definition.attention_head_count,
            mix_cfn_expansion_ratio=definition.mix_cfn_expansion_ratio,
            name=name,
            **kwargs,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'branch_height': self.branch_height,
            'branch_width': self.branch_width,
            'branch_depth': self.branch_depth,
            'transformer_count': self.transformer_count,
            'attention_window_size': self.attention_window_size,
            'attention_head_count': self.attention_head_count,
            'mix_cfn_expansion_ratio': self.mix_cfn_expansion_ratio,
            'name': self.name,
        })
        return config
