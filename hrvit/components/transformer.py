from keras import layers
from tensorflow import Tensor
from typing import Any, Optional

from .cross_shaped_attention import CrossShapedAttention
from .diversity_enhanced_shortcut import DiversityEnhancedShortcut
from .mix_cfn import MixCFN


class Transformer(layers.Layer):
    """
    Transformer layer.

    Args:
        attention_window_size: Size of the attention windows.
        attention_head_count: Number of attention heads.
        name: Name of the layer.
    """

    def __init__(
        self,
        attention_window_size: int = 1,
        attention_head_count: int = 2,
        mix_cfn_expansion_ratio: int = 4,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.attention = CrossShapedAttention(
            window_size=attention_window_size,
            head_count=attention_head_count,
        )
        self.mix_cfn = MixCFN(expansion_ratio=mix_cfn_expansion_ratio)
        self.diversity_enhanced_shortcut = DiversityEnhancedShortcut()
        self.layer_normalization_attention = layers.LayerNormalization()
        self.layer_normalization_mix_cfn = layers.LayerNormalization()

    def call(self, inputs: Tensor) -> Tensor:
        """
        Input shape: (batch_size, height, width, token_depth)
        Output shape: (batch_size, height, width, token_depth)
        """
        attention_residual = inputs
        des = self.diversity_enhanced_shortcut(inputs)
        attention_input = self.layer_normalization_attention(inputs)
        context = self.attention(attention_input) + des + attention_residual
        mix_cfn_residual = context
        mix_cfn_input = self.layer_normalization_mix_cfn(context)
        output = self.mix_cfn(mix_cfn_input) + mix_cfn_residual
        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'attention_window_size': self.attention.window_size,
            'attention_head_count': self.attention.head_count,
            'name': self.name,
        })
        return config
