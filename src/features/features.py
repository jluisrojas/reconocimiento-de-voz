import abc

"""
Clase de la cual heredean las clases para extraccion de features,
esta extraccion de features pueden ser de Histograma, MEL, etc.
"""
class ExtractorFeatures:
	"""
	Metodo que regresa un diccionario con los parametros del extractor de 
	features
	"""
	@abc.abstractmethod
	def get_config(self):
		pass
