import tensorflow as tf

from keras import layers
from tensorflow import Tensor
from typing import Any, Optional


class CrossResolutionFusion(layers.Layer):
    """
    The HRViT cross resolution fusion module. Combines tensors of differing
    resolutions and depths with channel matching, up-scaling, and down-sampling.

    Args:
        output_height: the height of the output block
        output_width: the width of the output block.
        output_depth: the depth of the output block (number of channels).
        name: Name of the layer.
    """

    def __init__(
        self,
        output_height: int = 224,
        output_width: int = 224,
        output_depth: int = 32,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.output_height = output_height
        self.output_width = output_width
        self.output_depth = output_depth
        self.activation_layer = layers.Activation(tf.keras.activations.gelu)

    def build(self, input_shape: tuple[tuple[int, int, int, int], ...]):
        self.input_operations = []
        for shape in input_shape:
            _batch_size, height, _width, depth = shape
            input_operations = []
            if height < self.output_height:
                scale = self.output_height // height
                input_operations.extend([
                    layers.Conv2D(
                        filters=self.output_depth,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        kernel_initializer="he_normal",
                        use_bias=True,
                    ),
                    layers.BatchNormalization(),
                    layers.UpSampling2D(size=scale, interpolation='nearest'),
                ])
            elif height > self.output_height:
                scale = height // self.output_height
                input_operations.extend([
                    layers.DepthwiseConv2D(
                        kernel_size=scale + 1,
                        strides=scale,
                        padding='same',
                        kernel_initializer="he_normal",
                        use_bias=False,
                    ),
                    layers.BatchNormalization(),
                    layers.Conv2D(
                        filters=self.output_depth,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        kernel_initializer="he_normal",
                        use_bias=True,
                    ),
                    layers.BatchNormalization(),
                ])
            elif depth != self.output_depth:
                input_operations.extend([
                    layers.Conv2D(
                        filters=self.output_depth,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        kernel_initializer="he_normal",
                        use_bias=True,
                    ),
                    layers.BatchNormalization(),
                ])
            self.input_operations.append(input_operations)

    def call(self, inputs: tuple[Tensor, ...]) -> Tensor:
        """
        Input shapes: N x (batch_size, height, width, token_depth)
        Output shape: (batch_size, height, width, token_depth)
        """
        preprocessed_inputs = []
        for input_operations, x in zip(self.input_operations, inputs):
            for operation in input_operations:
                x = operation(x)
            preprocessed_inputs.append(x)
        fused_inputs = tf.add_n(preprocessed_inputs)
        output = self.activation_layer(fused_inputs)
        return output

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'output_height': self.output_height,
            'output_width': self.output_width,
            'output_depth': self.output_depth,
            'name': self.name,
        })
        return config
