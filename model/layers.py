import tensorflow as tf
from tensorflow.keras import layers

"""
TODO:
    - Crear layer para obtener mask
    - Crear conv con mask
"""

"""
Capa que calcula el mask del input, esto solo funciona con el
siguiente shape: [batch, frames, rows, cols].
Y regresa: [batch, frames] de tipo booleano con el mask a los
frames
"""
class ObtenerMask(layers.Layer):
    def call(self, inputs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        shape = tf.shape(inputs)

        reshaped_input = tf.reshape(inputs, [shape[0], shape[1], -1])
        max_input = tf.math.reduce_max(reshaped_input, 2)

        mask = tf.not_equal(max_input, tf.zeros([shape[0], shape[1]],
            dtype=tf.float32))

        return mask
