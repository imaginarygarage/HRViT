import tensorflow as tf

from keras import layers
from tensorflow import Tensor
from typing import Any, Optional


class MixCFN(layers.Layer):
    """
    MixCFN. A mixed-scale convolutional FFN.

    Args:
        expansion_ratio: Number of hidden features, as a multiple of input
            features.
        name: Name of the layer.
    """

    def __init__(
        self,
        expansion_ratio: int = 4,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.expansion_ratio = expansion_ratio
        self.large_convolution = layers.DepthwiseConv2D(
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            kernel_initializer="he_normal",
        )
        self.small_convolution = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            kernel_initializer="he_normal",
        )
        self.gelu_activation = layers.Activation(tf.keras.activations.gelu)

    def build(self, input_shape: tuple[int, int, int, int]):
        _batch_size, _height, _width, token_depth = input_shape
        self.expanded_token_depth = token_depth * self.expansion_ratio
        self.expansion_projection = layers.Dense(units=self.expanded_token_depth)
        self.contraction_projection = layers.Dense(units=token_depth)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Input shape: (batch_size, height, width, token_depth)
        Output shape: (batch_size, height, width, token_depth)
        """
        expanded = self.expansion_projection(inputs)
        small = self.small_convolution(expanded[:, :, :, :self.expanded_token_depth // 2])
        large = self.large_convolution(expanded[:, :, :, self.expanded_token_depth // 2:])
        mixed_scale = tf.concat([small, large], axis=-1)
        mixed_scale = self.gelu_activation(mixed_scale)
        output = self.contraction_projection(mixed_scale)
        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'expansion_ratio': self.expansion_ratio,
            'name': self.name,
        })
        return config
