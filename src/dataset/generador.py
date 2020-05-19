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
        max_labels = max(tamanos_labels)
        max_frames= max(tamanos_frames)
        #print(max_frames)

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
            #num_frames_d.append(tf.convert_to_tensor([num_frames]))
            pf = 403 / 101
            num_frames_d.append(tf.convert_to_tensor([int(num_frames // pf)]))
            #num_frames_d.append(tf.convert_to_tensor([101]))
            labels_d.append(labels)
        """

        def gen():
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
                labels = tf.pad(labels_list[i],[[0, max_labels-num_labels]], constant_values=-1)

                # concatena el dataset
                features_d.append(x)
                num_labels_d.append(tf.convert_to_tensor([num_labels]))
                pf = 403 / 101
                num_frames_d.append(tf.convert_to_tensor([int(num_frames // pf)]))
                #num_frames_d.append(tf.convert_to_tensor([101]))
                labels_d.append(labels)

                yield (x, (labels, tf.convert_to_tensor([num_labels]), tf.convert_to_tensor([num_frames])))

        """




        #print(num_frames_d)
       	#return x_dataset, y_dataset
        #print("Tamano del dataset {}".format(len(features_d)))
        #print("shape de los features {}".format(features_d[0].shape))

        #dataset2 = tf.data.Dataset.from_generator( gen,(tf.float32, (tf.int32, tf.int32, tf.int32)))
        #print(dataset2)
        #return dataset2
        return features_d, labels_d, num_labels_d, num_frames_d

