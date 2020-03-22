import tensorflow as tf

"""
Clase que guarda la informacion del vocabulario en espaniol,
se guarda el mapping de caracteres para crear el embedding
"""
class EspVocabulario:
	"""
	Constructor que inicializa los caracteres del vocabulario y crea
	el diccionario con el mapping
	Args:
		incluir_simbolos: booleano si incluir simbolos como , . !
	"""
	def __init__(self, incluir_simbolos=False):
		self.caracteres = [
			"a", "b",
			"c", "d",
			"e", "f",
			"g", "h",
			"i", "j",
			"k", "l",
			"m", "n", "ñ",
			"o", "p",
			"q", "r",
			"s", "t",
			"u", "v",
			"w", "x",
			"y", "z",
			" "
		]

		self.vocab = {}
		for i, c in enumerate(self.caracteres):
			self.vocab[c] = i

	"""
	Metodo de tipo call el cual convierte una cadena en un arreglo de numeros
	con el numero del caracter, en caso de que el caracter no este en el vocabulario
	se ignora
	Args:
		cadena: string con la cadena a convertir
	"""
	def __call__(self, cadena):
		res = []
		cadena = cadena.lower()
		for c in cadena:
			if c in self.vocab:
				res.append(self.vocab[c])

		return res

	"""
	Regresa informacion del vocabulario usado
	"""
	def get_config(self):
		config = {
			"vocabulario": self.vocab
		}
		return config
