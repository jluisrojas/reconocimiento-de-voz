import tensorflow as tf
from .features import ExtractorFeatures


class SpectrogramaFeatures(ExtractorFeatures):
    def __init__(self,
                 sl=44100,
                 stft_fl=0.02, stft_fs=0.01, stft_fft=None):
        self.stft_fl = stft_fl
        self.stft_fs = stft_fs
        self.stft_fft = stft_fft
        self.sl = sl

    def get_config(self):
        return {
            "stft_frame_length": self.stft_fl,
            "stft_frame_step": self.stft_fs,
            "stft_fft_lenght": self.stft_fft
        }

    def __call__(self, sampling_rate, signal):
        self.sl = sampling_rate
        spectrogram = self.obtener_stft(signal)

        return spectrogram


    """
    Normaliza la señal para que este en el rango de
    -1.0 y 1.0
    """
    def normalizar_s(self, data):
        gain = 1.0 / (tf.math.reduce_max(tf.math.abs(data)) + 1e-5)
        return (data * gain)

    """
    Standariza la señal...
    """
    def standarizar_s(self, data):
        mean = tf.math.reduce_mean(data)
        std = tf.math.reduce_std(data)
        dataS = (data - mean) / std
        return dataS

    """
    Obtiene el spectrograma de la señal del audio, esto se hace
    aplicando Short Time Fourier Transformation. Primero la
    señal se normaliza y despues se standariza
    """
    def obtener_stft(self, data):
        data = self.normalizar_s(data)
        stfts = tf.signal.stft(tf.reshape(data, [1,-1]), frame_length=int(self.sl*self.stft_fl),
                               frame_step=int(self.sl*self.stft_fs), fft_length=self.stft_fft)
        stfts = tf.cast(stfts, tf.float32)
        stfts = tf.math.abs(stfts)

        # Estabilizar el offset para evitar altos rangos dinamicos
        log_offset = 1e-6
        stfts = tf.math.log(stfts + log_offset)

        stfts = self.standarizar_s(stfts)

        stfts = stfts / (tf.reduce_max(tf.abs(stfts)))

        return stfts
