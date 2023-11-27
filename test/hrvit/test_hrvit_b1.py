import keras_flops
import tensorflow as tf

from hrvit.hrvit_b1 import HRViT_b1


def test_hrvit_b1():
    model = HRViT_b1()
    x = tf.random.uniform((1, 224, 224, 3))

    y = model(x)

    assert len(y) == 4
    assert y[0].shape == (1, 56, 56, 32)
    assert y[1].shape == (1, 28, 28, 64)
    assert y[2].shape == (1, 14, 14, 128)
    assert y[3].shape == (1, 7, 7, 256)
    assert keras_flops.get_flops(model) == 3549873452


def test_hrvit_b1_predict():
    model = HRViT_b1()
    x = tf.random.uniform((1, 224, 224, 3))

    y = model.predict(x)

    assert len(y) == 4
    assert y[0].shape == (1, 56, 56, 32)
    assert y[1].shape == (1, 28, 28, 64)
    assert y[2].shape == (1, 14, 14, 128)
    assert y[3].shape == (1, 7, 7, 256)
