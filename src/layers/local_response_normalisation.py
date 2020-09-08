import tensorflow as tf
from tensorflow.keras.layers import Layer


class LocalResponseNormalization(Layer):
    def __init__(self, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.lrn(inputs, bias=2, depth_radius=5, alpha=0.0001, beta=0.75)

    def get_config(self):
        return super().get_config()
