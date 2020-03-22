import abc

"""
Clase abstracta que define la implementacion del Pipeline de
entrenamiento
"""
class Pipeline:
	"""
	Metodo que hace "fit" del dataset al modelo
	Args:
		train_descrip: objeto de tipo DataDescripcion con la descripcion
					   del dataset de entrenamiento
		test_descrip: objeto de tipo DataDescripcion con la descripcion
					   del dataset de pruebas
		setup: diccionario con hyperparametros y setup de entrenamiento
	"""
	@abc.abstractmethod
	def fit(self, train_descrip, test_descrip, setup):
		pass

	"""
	Metodo que regresa la configuracion del pipeline
	"""
	@abc.abstractmethod
	def get_config(self):
		pass

