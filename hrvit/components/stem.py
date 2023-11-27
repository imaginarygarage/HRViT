from keras import layers
from typing import Optional


class Stem(layers.Layer):
    """
    Convolutional stem block. Performs the initial feature extraction prior to
    ViT layers.

    Args:
        halving_factor: Number of times the height and width of the input block
            is halved.
        output_channels: Number of channels in the output block.
        kernel_size: Kernel size.
        name: Name of the layer.
    """

    def __init__(
        self,
        halving_factor: int = 2,
        output_channels: int = 32,
        kernel_size: int = 3,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.halving_factor = halving_factor
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        filters = self.output_channels // 2 ** self.halving_factor
        self.layers = [
            layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=(1, 1),
                padding='same',
                kernel_initializer="he_normal",
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]
        for _ in range(self.halving_factor):
            filters *= 2
            self.layers.extend([
                layers.Conv2D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer="he_normal",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])

    def call(self, inputs):
        """
        Input shape: (batch_size, height, width, channels)
        Output shape: (batch_size, height / 2**halving_factor, width / 2**halving_factor, output_channels)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'halving_factor': self.halving_factor,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'name': self.name,
        })
        return config
