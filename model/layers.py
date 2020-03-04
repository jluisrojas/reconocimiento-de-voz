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
        shape = tf.shape(inputs)

        reshaped_input = tf.reshape(inputs, [shape[0], shape[1], -1])
        max_input = tf.math.reduce_max(reshaped_input, 2)

        mask = tf.not_equal(max_input, tf.zeros([shape[0], shape[1]],
            dtype=tf.float32))

        return mask

class MaskWrapper(layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super(MaskWrapper, self).__init__(layer, **kwargs)
        self.supports_masking = True

    def call(self, inputs):
        """
        NOTA: se supone que el call no tiene argumentos
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = traininginp
        """
        return self.layer.call(inputs)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
