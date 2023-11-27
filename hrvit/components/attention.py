import numpy as np
import tensorflow as tf

from keras import layers
from tensorflow import Tensor
from typing import Any, Optional


class Attention(layers.Layer):
    """
    Attention layer. Computes the context vector of a given query, key, and
    value.

    Args:
        padding_tokens: Number of padding tokens included in the input.
        name: Name of the layer.
    """

    def __init__(
        self,
        padding_tokens: int = 0,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.padding_tokens = padding_tokens
        self.padding_mask: Optional[Tensor] = None
        self.softmax = layers.Softmax(axis=-1)

    def build(self, input_shape: tuple[tuple[int, int, int, int, int], ...]):
        _batch_size, _window_count, head_count, window_token_count, token_depth_per_head = input_shape[0]
        self.attention_scalar = token_depth_per_head ** -0.5
        if self.padding_tokens > 0:
            self.padding_mask = np.zeros((1, 1, head_count, window_token_count, window_token_count))
            self.padding_mask[:, :, :, -self.padding_tokens:, :] = -float('inf')
            self.padding_mask[:, :, :, :, -self.padding_tokens:] = -float('inf')
            self.padding_mask = tf.constant(self.padding_mask, dtype=tf.float32)

    def call(self, inputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Input shapes: 3 x (batch_size, windows, heads, pixels per window, token depth per head)
        Output shape: (batch_size, windows, heads, pixels per window, token depth per head)
        """
        q, k, v = inputs
        attention = tf.linalg.matmul(a=q, b=k, transpose_b=True) * self.attention_scalar
        attention = self._mask_attention(attention)
        attention = self.softmax(attention)
        context = tf.linalg.matmul(a=attention, b=v)
        return context

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'padding_tokens': self.padding_tokens,
            'name': self.name,
        })
        return config

    def _mask_attention(self, attention: Tensor) -> Tensor:
        """
        Masks the attention matrix to prevent tokens from attending to padding
        tokens.
        """
        if self.padding_mask is not None:
            attention_window_to_mask = attention[:, -1:, :, :, :]
            masked_attention_window = attention_window_to_mask + self.padding_mask
            attention = tf.concat([attention[:, :-1, :, :, :], masked_attention_window], axis=1)
        return attention
