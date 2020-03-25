import tensorflow as tf
from .layers import ObtenerMask, MaskWrapper
from tensorflow.keras.layers import ConvLSTM2D, GRU, Input, Bidirectional
from tensorflow.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.keras.layers import Flatten, Dense, ReLU, Reshape, Conv2D
from tensorflow.keras import Model

def obtener_ds2(input_dim=(10, 250), num_convs=1, num_labels=27):
    input_tensor = Input(shape=input_dim, name="x")
    # x = ObtenerMask()(input_tensor)

    x = input_tensor

    for i in range(2):
        x = Conv2D(filters=32, 
                   kernel_size=(11, 11), 
                   strides=[2, 2],
                   padding="same",
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # x = TimeDistributed(Flatten())(x)
    x = Reshape([-1, 32*32])(x)
    for i in range(5):
        x = Bidirectional(GRU(units=512, activation="tanh", recurrent_activation="sigmoid", return_sequences=True), merge_mode="concat")(x)

    x = TimeDistributed(Dense(1024))(x)
    x = ReLU()(x)
    x = TimeDistributed(Dense(num_labels+1))(x)

    output_tensor = x

    model = Model(input_tensor, output_tensor, name="DeepSpeech2")

    return model
