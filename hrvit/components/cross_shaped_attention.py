import tensorflow as tf

from keras import layers
from tensorflow import Tensor
from typing import Any, Optional

from .attention import Attention
from .hard_swish import HardSwish


class CrossShapedAttention(layers.Layer):
    """
    Cross shaped attention layer.

    Args:
        window_size: width of the vertical attention windows, height of the
            horizontal attention windows.
        name: Name of the layer.
    """

    def __init__(
        self,
        window_size: int = 0,
        head_count: int = 2,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.window_size = window_size
        self.head_count = head_count
        self.inductive_bias_hard_swish = HardSwish()
        self.output_hard_swish = HardSwish()
        self.output_dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape: tuple[int, int, int, int]):
        _batch_size, self.height, self.width, self.token_depth = input_shape
        self.horizontal_window_count = (self.height + self.window_size - 1) // self.window_size
        self.vertical_window_count = (self.width + self.window_size - 1) // self.window_size
        self.padded_height = self.horizontal_window_count * self.window_size
        self.padded_width = self.vertical_window_count * self.window_size
        self.horizontal_padding = self.window_size * self.horizontal_window_count - self.height
        self.vertical_padding = self.window_size * self.vertical_window_count - self.width
        self.horizontal_attention = Attention(padding_tokens=self.horizontal_padding)
        self.vertical_attention = Attention(padding_tokens=self.vertical_padding)
        self.qkv_projection = layers.Dense(units=self.token_depth * 2)
        self.inductive_bias_convolution = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            kernel_initializer="he_normal",
        )
        self.output_projection = layers.Dense(units=self.token_depth)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Input shape: (batch_size, height, width, token depth)
        Output shape: (batch_size, height, width, token depth)
        """
        q, k, v = self._construct_qkv(inputs)
        qh, kh, vh = [self._reshape_2d_for_horizontal_attention(x) for x in (q, k, v)]
        qv, kv, vv = [self._reshape_2d_for_vertical_attention(x) for x in (q, k, v)]
        context_h = self.horizontal_attention((qh, kh, vh))
        context_v = self.vertical_attention((qv, kv, vv))
        context = tf.concat(
            [self._reshape_horizontal_attention_to_2d(context_h), self._reshape_vertical_attention_to_2d(context_v)],
            axis=-1
        )
        context += self._auxiliary_inductive_bias(v)
        context = self.output_hard_swish(context)
        context = self.output_projection(context)
        context = self.output_dropout(context)
        return context

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'window_size': self.window_size,
            'head_count': self.head_count,
            'name': self.name,
        })
        return config

    def _auxiliary_inductive_bias(self, values: Tensor) -> Tensor:
        """
        Build the auxiliary inductive bias tensor.
        """
        hard_swished_values = self.inductive_bias_hard_swish(values)
        auxiliary_inductive_bias = self.inductive_bias_convolution(hard_swished_values)
        return auxiliary_inductive_bias

    def _construct_qkv(self, token_block: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Build the query, key, and value tensors from the input tokens.
        """
        qkv = self.qkv_projection(token_block)
        q, k = tf.split(qkv, num_or_size_splits=2, axis=-1)
        v = k
        return q, k, v

    def _reshape_2d_for_horizontal_attention(self, input_block: Tensor) -> Tensor:
        """
        Reshape the given tensor into a set of horizontal attention windows.
        input shape: (batch_size, height, width, token_depth)
        output shape: (batch_size, window_count, head_count, pixels_per_window, token_depth_per_head)
        """
        _batch_size, _height, width, token_depth = input_block.shape
        horizontal_block = input_block[:, :, :, :token_depth // 2]
        padded_block = tf.pad(horizontal_block, [[0, 0], [0, self.horizontal_padding], [0, 0], [0, 0]])
        reshaped_block = layers.Reshape(
            target_shape=(self.horizontal_window_count, self.window_size, width, self.head_count, -1)
        )(padded_block)
        permuted_block = tf.transpose(reshaped_block, perm=(0, 1, 4, 2, 3, 5))
        flattened_block = layers.Reshape(
            target_shape=(self.horizontal_window_count, self.head_count, self.window_size * width, -1)
        )(permuted_block)
        return flattened_block

    def _reshape_2d_for_vertical_attention(self, input_block: Tensor) -> Tensor:
        """
        Reshape the given tensor into a set of vertical attention windows.
        input shape: (batch_size, height, width, token_depth)
        output shape: (batch_size, window_count, head_count, pixels_per_window, token_depth_per_head)
        """
        _batch_size, height, _width, token_depth = input_block.shape
        vertical_block = input_block[:, :, :, token_depth // 2:]
        padded_block = tf.pad(vertical_block, [[0, 0], [0, 0], [0, self.vertical_padding], [0, 0]])
        reshaped_block = layers.Reshape(
            target_shape=(height, self.vertical_window_count, self.window_size, self.head_count, -1)
        )(padded_block)
        permuted_block = tf.transpose(reshaped_block, perm=(0, 2, 4, 1, 3, 5))
        flattened_block = layers.Reshape(
            target_shape=(self.vertical_window_count, self.head_count, self.window_size * height, -1)
        )(permuted_block)
        return flattened_block

    def _reshape_horizontal_attention_to_2d(self, input_block: Tensor) -> Tensor:
        """
        Reshape the given tensor from a set of horizontal attention windows.
        input shape: (batch_size, window_count, head_count, pixels_per_window, token_depth_per_head)
        output shape: (batch_size, height, width, token_depth)
        """
        _batch_size, window_count, head_count, _pixels_per_window, _token_depth_per_head = input_block.shape
        unflattened_block = layers.Reshape(
            target_shape=(window_count, head_count, self.window_size, self.width, -1)
        )(input_block)
        unpermuted_block = tf.transpose(unflattened_block, perm=(0, 1, 3, 4, 2, 5))
        unreshaped_block = layers.Reshape(
            target_shape=(self.padded_height, self.width, self.token_depth // 2)
        )(unpermuted_block)
        unpadded_block = unreshaped_block[:, :self.height, :self.width, :]
        return unpadded_block

    def _reshape_vertical_attention_to_2d(self, input_block: Tensor) -> Tensor:
        """
        Reshape the given tensor from a set of vertical attention windows.
        input shape: (batch_size, window_count, head_count, pixels_per_window, token_depth_per_head)
        output shape: (batch_size, height, width, token_depth)
        """
        _batch_size, window_count, head_count, _pixels_per_window, _token_depth_per_head = input_block.shape
        unflattened_block = layers.Reshape(
            target_shape=(window_count, head_count, self.window_size, self.height, -1)
        )(input_block)
        unpermuted_block = tf.transpose(unflattened_block, perm=(0, 1, 3, 2, 4, 5))
        unreshaped_block = layers.Reshape(
            target_shape=(self.height, self.padded_width, self.token_depth // 2)
        )(unpermuted_block)
        unpadded_block = unreshaped_block[:, :self.height, :self.width, :]
        return unpadded_block
