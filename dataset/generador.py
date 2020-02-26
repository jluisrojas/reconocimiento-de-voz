import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import pandas as pd

"""
Clase encargada de generar el dataset desde leer archivos
hasta preparacion para Tensorflow Dataset. 
El proceso es el siguiente:
    1. Leer rutas y la frases del csv
    2. Por cada entrada en el csv
        - Leer wav
        - Aplicar STFT
        - Nomalizar, standarizar, etc.
        - Convertir a frames
        - Convertir frase a representacion vectorial
        - Convertir a una entrada de Tensorflow Dataset
    3. Guardar dataset
    4. Regresar dataset
"""
class GeneradorDataset():
    """
    
    """
    def __init__(self, features_extractor, fl=10, fs=10):
        self.features_extractor = features_extractor
        self.fl = fl
        self.fs = fs

    """
    Lee el archivo .wav convierte la senal en
    en un tensor de tipo float32
    """
    def leer_wav(self, archivo_wav):
        sampling_rate, data = wavfile.read(archivo_wav)
        data = tf.convert_to_tensor(data)
        data = tf.cast(data, tf.float32)
        return sampling_rate, data

    def leer_csv(self, archivo_csv):
        df = pd.read_csv(archivo_csv)
        return df#.iloc[0]

    """
    Genera el dataset de una cierta distribucion
    """
    def generar_distribucion(self, ruta, distribucion, sub_ruta=""):
        dataset = None
        df = self.leer_csv(ruta + distribucion + ".csv")

        for indice, renglon in df.iterrows():
            sl, data = self.leer_wav(ruta + sub_ruta + renglon["path"])

            # Obtine features del feature extractor
            features = self.features_extractor(sl, data)

            # Divide el spectrgrama en frames
            frames = tf.signal.frame(features, self.fl, self.fs, axis=1, pad_end=True)
            print(frames)

            if dataset == None:
                dataset = frames
            else:
                dataset = tf.concat([dataset, frames], 0)

        return dataset

