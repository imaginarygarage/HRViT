import numpy as np
import pytest
import tensorflow as tf

from hrvit.components.attention import Attention


@pytest.mark.parametrize('input_shape, padding_tokens, expected_output_shape', [
    ((1, 5, 4, 12, 5), 0, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 1, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 2, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 3, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 4, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 5, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 6, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 7, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 8, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 9, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 10, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 11, (1, 5, 4, 12, 5)),
    ((1, 5, 4, 12, 5), 12, (1, 5, 4, 12, 5)),
    ((3, 2, 6, 11, 9), 0, (3, 2, 6, 11, 9)),
    ((3, 2, 6, 11, 9), 7, (3, 2, 6, 11, 9)),
])
def test_attention_produces_expected_shape(
    input_shape: tuple[int, int, int, int],
    padding_tokens: int,
    expected_output_shape: tuple[int, int, int],
):
    attention = Attention(padding_tokens)

    q, k, v = tf.random.uniform(input_shape), tf.random.uniform(input_shape), tf.random.uniform(input_shape)
    context = attention((q, k, v))

    assert context.shape == expected_output_shape
    if padding_tokens > 0:
        assert attention.padding_mask is not None
        assert attention.padding_mask.shape == (1, 1, input_shape[2], input_shape[3], input_shape[3])
        assert np.all(attention.padding_mask[:, :, :, :-padding_tokens, :-padding_tokens].numpy() == 0.0)
        assert np.all(attention.padding_mask[:, :, :, -padding_tokens:, -padding_tokens:].numpy() == -float('inf'))
        assert np.all(np.isnan(context[:, -1, :, -padding_tokens:, :].numpy()))
        assert np.all(~np.isnan(context[:, -1, :, :-padding_tokens, :].numpy()))
        assert np.all(~np.isnan(context[:, :-1, :, :, :].numpy()))
    else:
        assert attention.padding_mask is None
        assert np.all(~np.isnan(context.numpy()))


def test_attention_value_with_padded_input_produces_expected_context():
    attention = Attention(padding_tokens=2)

    q = k = v = tf.constant([[[[
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [7.0, 7.0, 7.0],
        [7.0, 7.0, 7.0],
    ]]]], dtype=tf.float32)
    context = attention((q, k, v))

    expected_context = np.array([[[[
        [1.1806947, 2.0, 2.8193054],
        [2.8193054, 2.0, 1.1806947],
        [float('nan'), float('nan'), float('nan')],
        [float('nan'), float('nan'), float('nan')],
    ]]]], dtype=np.float32)

    assert np.allclose(context.numpy()[:, :, :, :2, :], expected_context[:, :, :, :2, :])
    assert np.all(np.isnan(context.numpy()[:, :, :, 2:, :]))
    assert np.all(np.isnan(expected_context[:, :, :, 2:, :]))
