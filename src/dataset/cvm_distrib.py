from .dataset_distrib import DataDistrib
from .generador import GeneradorDataset
import tensorflow as tf

class CVMDistrib(DataDistrib):
    def __init__(self, features, vocabulario):
        self.features = features
        self.vocabulario = vocabulario

    def __call__(self, data_description):
        distribucion = data_description.distribucion
        tamano = data_description.tamano

        # SE CHECARIA SI EL DATASET YA SE GENERO UNA VEZ
        # SI ESE ES EL CASO, SE CARGA DE TF RECORD
        generador = GeneradorDataset(spectrograma, vocabulario, fl=10, fs=10)
        features, labels, num_labels, num_frames = generador.generar_distribucion("dataset/common-voice/es/",
                distribucion, sub_ruta="clips/")
        dataset = tf.data.Dataset.from_tensor_slices((features, (labels,
            num_labels, num_frames)))

        return dataset

    def get_config(self):
        pass
