import tensorflow as tf

from keras import layers
from tensorflow import Tensor
from typing import Any, Optional


class HardSwish(layers.Layer):
    """
    Hard swish activation function.

    Args:
        name: Name of the layer.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        hard_swished = inputs * tf.keras.activations.hard_sigmoid(inputs)
        return hard_swished

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'name': self.name,
        })
        return config
