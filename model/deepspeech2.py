import tensorflow as tf
from .layers import ObtenerMask, MaskWrapper
from tensorflow.keras.layers import ConvLSTM2D, GRU, Input, Bidirectional
from tensorflow.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.keras.layers import Flatten, Dense, ReLU
from tensorflow.keras import Model

def obtener_ds2(input_dim=(10, 250), num_convs=1, num_labels=27):
    input_tensor = Input(shape=input_dim, name="x")
    x = ObtenerMask()(input_tensor)

    for i in range(num_convs):
        x = ConvLSTM2D(8, (4, 4), padding="valid", return_sequences=True,
                data_format="channels_last")(x)
        x = BatchNormalization()(x)

    x = TimeDistributed(MaskWrapper(Flatten()))(x)
    for i in range(6):
        #pass
        x = Bidirectional(GRU(32, return_sequences=True))(x)

    x = TimeDistributed(Dense(64))(x)
    x = ReLU()(x)
    x = TimeDistributed(Dense(num_labels))(x)

    output_tensor = x

    model = Model(input_tensor, output_tensor, name="DeepSpeech2")

    return model
