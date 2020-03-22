import abc

"""
Clase que describe la distribucion del dataset para el entrenamiento.
La descripcion contine los siguientes atributos:
	distribucion: train, test, dev
	tamano: None si es todo el dataset, un entero si se utiliza un subset del dataset
"""
class DataDescripcion:
	"""
	Constructor del objeto donde se envian los atributos de la descripcion
	Args:
		distribucion: string ej. "train", "dev", "test"
		tamano: None por defecto, si no es el tamano del subdataset
	"""
	def __init__(self,
			distribucion="train",
			tamano=None):
		self.distribucion = distribucion
		self.tamano = tamano

	"""
	Metodo que regresa los atributos de la descripcion en un diccionario
	"""
	def get_config(self):
		return {
			"distribucion": self.distribucion,
			"tamano": self.tamano
		}


"""
Clase abstracta para implementar una distribucion del dataset, esto es para poder cargar
los datos dependiendo del origen de donde se quieran los datos
"""
class DataDistrib:
	"""
	Metodo para regresar la configuracion de la distribucion
	"""
	@abc.abstractmethod
	def get_config(self):
		pass
