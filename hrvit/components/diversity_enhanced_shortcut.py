import math
import tensorflow as tf

from keras import layers
from tensorflow import Tensor
from typing import Any, Optional


class DiversityEnhancedShortcut(layers.Layer):
    """
    DES. A diversity enhanced shortcut.

    Args:
        name: Name of the layer.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gelu_activation = layers.Activation(tf.keras.activations.gelu)

    def build(self, input_shape: tuple[int, int, int, int]):
        _batch_size, _height, _width, token_depth = input_shape
        self.k, self.p = self._decompose(token_depth)
        self.project_right = layers.Dense(units=self.p)
        self.project_left = layers.Dense(units=self.k)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Input shape: (batch_size, height, width, token_depth)
        Output shape: (batch_size, height, width, token_depth)
        """
        project_right_input = layers.Reshape((inputs.shape[1], inputs.shape[2], self.k, self.p))(inputs)
        projected_right = self.project_right(project_right_input)
        projected_right = self.gelu_activation(projected_right)
        project_left_input = tf.transpose(projected_right, perm=[0, 1, 2, 4, 3])
        projected_left = self.project_left(project_left_input)
        output = tf.transpose(projected_left, perm=[0, 1, 2, 4, 3])
        output = layers.Reshape((inputs.shape[1], inputs.shape[2], inputs.shape[3]))(output)
        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'name': self.name,
        })
        return config

    def _decompose(self, token_depth: int) -> tuple[int, int]:
        exponent = int(math.log2(token_depth))
        p = 2 ** (exponent - exponent // 2)
        k = token_depth // p
        return k, p
