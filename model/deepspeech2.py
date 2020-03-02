import tensorflow as tf
from layers import ObtenerMask
from tensorflow.keras.layers import ConvLSTM2D, GRU, Input
from tensorflow.keras.layers import BatchNormalization, ReLu
from tensorflow.keras import Model

def obtener_ds2(input_dim=(10, 250), num_convs=1):
    input_tensor = Input([None, input_dim[0], input_dim[1]], name="x")
    for i in range(num_convs):
        x = ConvLSTM2D(5, (3, 3))(x)
        x = BatchNormalization()(x)

    for in range(6):


    output_tensor = x

    model = Model(input_tensor, output_tensor, name="DeepSpeech2")

    return model
