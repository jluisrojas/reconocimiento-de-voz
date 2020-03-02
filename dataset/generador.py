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
        tamanos_frames = []
        dataset = []
        masks = []
        df = self.leer_csv(ruta + distribucion + ".csv")

        for indice, renglon in df.iterrows():
            sl, data = self.leer_wav(ruta + sub_ruta + renglon["path"])

            # Obtine features del feature extractor
            features = self.features_extractor(sl, data)

            # Divide el spectrgrama en frames
            frames = tf.signal.frame(features, self.fl, self.fs, axis=1, pad_end=True)
            num_frames = frames.shape[1]

            tamanos_frames.append(num_frames)
            dataset.append(frames)
            # Crea el mask del input
            masks.append(tf.ones([1, num_frames]))

        # Obtiene el numero mayor de frames en el dataset de esta 
        # manera se realiza el padding para entrenamiento
        padding = max(tamanos_frames)
        padded_dataset = []
        padded_masks = []

        # Padea todos los elementos del dataset
        for i, num_frames in enumerate(tamanos_frames):
            # Agrega padding al input
            paddings = [[0,0], [0, padding - num_frames], [0,0], [0,0]]
            frames = tf.pad(dataset[i], paddings, "CONSTANT")
            frames = tf.expand_dims(frames, -1)

            padded_dataset.append(frames)

            """
            # Agrega padding al mask y lo hace booleano
            #mask = tf.pad(masks[i], [[0,0], [0, padding - num_frames]], "CONSTANT")
            #mask = tf.math.equal(mask, tf.ones([1, padding]))

            s = frames.shape

            mask = tf.reshape(frames, [s[-4], s[-3], s[-1]*s[-2]])
            mask = tf.math.reduce_max(mask, 2)

            mask2 = tf.not_equal(mask, tf.zeros([s[-4],s[-3]], dtype=tf.float32))
            mask = tf.boolean_mask(mask, mask2)

            mask = tf.math.reduce_sum(tf.ones_like(mask), 0)
            print(mask)

            s = frames.shape
            print(s)
            mask = tf.keras.layers.Masking(mask_value=0.0)(tf.reshape(frames,
                [s[-4], s[-3], s[-1]*s[-2]]))
            print(mask.shape)
            padded_masks.append(mask)
            """

        tensor_dataset = tf.concat(padded_dataset, 0)
        #tensor_masks = tf.concat(padded_masks, 0)
        #print(tensor_masks)

        return tensor_dataset

