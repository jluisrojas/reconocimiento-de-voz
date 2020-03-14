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
    def __init__(self, features_extractor, vocabulario, fl=10, fs=10):
        self.features_extractor = features_extractor
        self.vocabulario = vocabulario
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

    def leer_csv(self, archivo_csv, tamano=None):
        df = pd.read_csv(archivo_csv)
        if tamano is not None:
            return df.head(tamano)
        else:
            return df

    """
    Genera el dataset de una cierta distribucion
    """
    def generar_distribucion(self, ruta, distribucion, sub_ruta="", tamano=None):
        # Labels
        tamanos_labels = []
        labels_list = []

        # Features
        tamanos_frames = []
        dataset = []
        masks = []
        df = self.leer_csv(ruta + distribucion + ".csv", tamano=tamano)

        for indice, renglon in df.iterrows():
            # Precicciones
            cadena = renglon["sentence"]
            logits = tf.convert_to_tensor(self.vocabulario(cadena))
            labels = logits
            #labels = tf.expand_dims(logits, -1)

            tamanos_labels.append(labels.shape[0])
            labels_list.append(labels)

            # Features
            sl, data = self.leer_wav(ruta + sub_ruta + renglon["path"])

            # Obtine features del feature extractor
            features = self.features_extractor(sl, data)

            if self.fl > 0:
                # Divide el spectrgrama en frames
                frames = tf.signal.frame(features, self.fl, self.fs, axis=1, pad_end=True)
            else:
                frames = features

            num_frames = frames.shape[1]

            tamanos_frames.append(num_frames)
            dataset.append(frames)
            # Crea el mask del input
            masks.append(tf.ones([1, num_frames]))

        # Obtiene el numero mayor de frames y labels en el dataset de esta 
        # manera se realiza el padding para entrenamiento
        max_labels = max(tamanos_labels) + 1
        max_frames= max(tamanos_frames)

        features_d = []
        labels_d = []
        num_labels_d = []
        num_frames_d = []

        # Padea todos los elementos del dataset
        for i, num_frames in enumerate(tamanos_frames):
            # Agrega padding a los features
            if self.fl > 0:
                paddings = [[0,0], [0, max_frames- num_frames], [0,0], [0,0]]
            else:
                paddings = [[0,0], [0, max_frames- num_frames], [0,0]]

            frames = tf.pad(dataset[i], paddings, "CONSTANT")
            frames = tf.expand_dims(frames, -1)
            x = tf.squeeze(frames, axis=0)

            # Agrega padding a los labels
            num_labels = tamanos_labels[i]
            #labels = tf.pad(labels_list[i],[[0, max_labels-num_labels], [0,
            #    0]], constant_values=-1)
            labels = tf.pad(labels_list[i],[[0, max_labels-num_labels]], constant_values=-1)

            # concatena el dataset
            features_d.append(x)
            num_labels_d.append(tf.convert_to_tensor([num_labels]))
            num_frames_d.append(tf.convert_to_tensor([num_frames]))
            labels_d.append(labels)


            #print(y_true)
            """
            print("features: " + str(frames.shape))
            print("labels: " + str(labels.shape))
            print("label_length: " + str(num_labels))
            print("frames_length: " + str(num_frames))
            """

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

        #return x_dataset, y_dataset
        return features_d, labels_d, num_labels_d, num_frames_d

