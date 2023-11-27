from keras import layers
from tensorflow import Tensor
from typing import Any, Optional


class PatchEmbed(layers.Layer):
    """
    Convolutional patch embedding. Tokenizes the input image for consumption by
    ViT layers.

    Args:
        token_depth: dimensionality of each token.
        token_receptive_field: Receptive field of each token.
        name: Name of the layer.
    """

    def __init__(
        self,
        token_depth: int = 32,
        token_receptive_field: int = 3,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.token_depth = token_depth
        self.token_receptive_field = token_receptive_field
        self.linear_projection = layers.Conv2D(
            filters=self.token_depth,
            kernel_size=1,
            strides=(1, 1),
            padding='same',
            kernel_initializer="he_normal",
        )
        self.depthwise_convolution = layers.DepthwiseConv2D(
            kernel_size=self.token_receptive_field,
            strides=(1, 1),
            padding='same',
            kernel_initializer="he_normal",
        )
        self.layer_normalization = layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Input shape: (batch_size, height, width, channels)
        Output shape: (batch_size, height, width, token_depth)
        """
        x = inputs
        x = self.linear_projection(x)
        x = self.depthwise_convolution(x)
        x = self.layer_normalization(x)
        return x

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'token_depth': self.token_depth,
            'token_receptive_field': self.token_receptive_field,
            'name': self.name,
        })
        return config
