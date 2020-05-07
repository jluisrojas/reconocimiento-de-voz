from .dataset_distrib import DataDistrib
from .generador import GeneradorDataset
import tensorflow as tf

"""
Common Voice Mozilla Distribution
Genera la distribucion del dataset de common-voice para 
entrenamiento durante el pipeline
"""
class CVMDistrib(DataDistrib):
	"""
	Constructor del objeto de la distribucion
	Argumentos:
		features: Objeto de tipo ExtractorFeatures
		vocabulario: Objeto con informacio del vocabulario usado
		frame_length: la longitud del frame de features en milisegundos
		frame_step: la longitud del paso de frame a frame en milisegundos
		root_path: el path de donde se obtiene los archivos del dataset
	"""
	def __init__(self, features=None, vocabulario=None,
				 frame_length=10,
				 frame_step=10,
				 root_path="src/dataset/common-voice/es/"):
		self.features = features
		self.vocabulario = vocabulario
		self.fl = frame_length
		self.fs = frame_step
		self.root_path = root_path

	"""
	Metodo que regresa el dataset de la distribucion dependiendo del
	DataDescription, aqui se puede cargar o generar dependiendo.
	Args:
		data_description: objeto de tipo DataDescription con info del dataset a generar
	"""
	def __call__(self, data_description):
		distribucion = data_description.distribucion
		tamano = data_description.tamano

		# SE CHECARIA SI EL DATASET YA SE GENERO UNA VEZ
		# SI ESE ES EL CASO, SE CARGA DE TF RECORD
		generador = GeneradorDataset(self.features, self.vocabulario, fl=self.fl, fs=self.fs)
		#features, labels, num_labels, num_frames = generador.generar_distribucion(self.root_path, distribucion, sub_ruta="clips/", tamano=tamano)

		# Convierte los tensors a tf.data.Dataset
		#dataset = tf.data.Dataset.from_tensor_slices((features, (labels, num_labels, num_frames)))
		dataset = generador.generar_distribucion(self.root_path, distribucion, sub_ruta="clips/", tamano=tamano)

		return dataset

	"""
	Regresa la configuracion del objeto
	"""
	def get_config(self):
		return {
			"frame_length": self.fl,
			"frame_step": self.fs,
			"root_path": self.root_path
		}
