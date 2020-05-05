import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

from .pipeline import Pipeline
#from .. import ctc
from ..ctc import get_loss
from ..metricas import wer

"""
Se define el pipeline de entrenamiento para Deep Speech 2, considerando:
	- CTC loss
"""

class DS2Pipeline(Pipeline):

	def memory(self):
		import os
		import psutil
		pid = os.getpid()
		py = psutil.Process(pid)
		memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
		print('memory use:', memoryUse)


	"""
	Constructorydel pipeline de ds2
	Args:
		model: Keras Model que se va usar para entrenarlos
		features: objeto de tipo FeatureExtractor para obtener features del dataset
		vocabulario: vocabulario del dataset
		dataset_distrib: objeto de tipo DataDistrib que define la distribucion del dataset
	"""
	def __init__(self, model=None, features=None, vocabulario=None, 
			dataset_distrib=None, nombre="DS2"):
		self.nombre = nombre
		self.model = model
		self.features = features
		self.vocabulario = vocabulario
		self.dataset_distrib = dataset_distrib

		self.logs_path = "./training_logs/"


	def get_config(self):
		return {
			"nombre": self.nombre,
			"descripcion": "Deep Speech 2 Pipeline",
			"features": self.features.get_config(),
			"vocabulario": self.vocabulario.get_config(),
			"dataset_dsitrib": self.dataset_distrib.get_config()
		}

	@tf.function
	def predict(self, x, y, training=True):
			return self.model(x, training=training)

	@tf.function
	def loss(self, x, y, training):
			y_ = self.predict(x, y, training=True)

			return y_, get_loss(y, y_)

	@tf.function
	def grad(self, inputs, targets):
			with tf.GradientTape() as tape:
				y_, loss_value = self.loss(inputs, targets, training=True)

			return y_, loss_value, tape.gradient(loss_value, self.model.trainable_variables)
	"""
	Inicia el entrenamiento del modelo con las descripciones del dataset
	Args:
		train_descrip: objeto de tipo DataDescripcion con la descripcion del
					   dataset de entrenamiento
		test_descirp: objeto de tipo DataDescripcion con las descripcion del
					  dataset de pruebas
		setup: diccionario con hyperparametros y setup de entrenamiento
	"""
	def fit(self,  train_descrip, test_descrip, setup):
		with open(self.logs_path + self.nombre + "/" + "training.json", "w") as json_file:
			json.dump({
					"setup": setup,
					"train_descrip": train_descrip.get_config(),
					"test_descrip": test_descrip.get_config()
				}, json_file, indent=4)

		print("[INFO] Cargando distribuciones del datasetpipeline")
		train = self.dataset_distrib(train_descrip)
		#test = self.dataset_distrib(test_descrip)

		print(train)

		lr = setup["learning_rate"]
		bs = setup["batch_size"]
		epochs = setup["epochs"]
		i_epoch = setup["initial_epoch"]

		optimizer = Adam(learning_rate=3e-4)
		#optimizer = RMSprop()

		train = train.batch(bs)
		#test  = test.batch(bs)

		self.train(optimizer, train, epochs)


	def train(self, optimizer, train, epochs):
		columnas = ["epoch", "loss", "WER"]
		logs = { }
		for c in columnas:
			logs[c] = []

		for epoch in range(epochs):
			logs["epoch"].append(epoch)
			print("Iniciando epoch: {}".format(epoch))
			self.memory()

			epoch_loss, epoch_wer = self.train_epoch(optimizer, train, epoch)

			if not epoch_loss == None:
				loss_mean = tf.reduce_mean(epoch_loss).numpy()
				wer_mean = tf.reduce_mean(epoch_wer).numpy()
				logs["loss"].append(loss_mean)
				logs["WER"].append(wer_mean)

				print("Epoch {:03d}: Loss: {:.3f} WER: {}".format(epoch, loss_mean, wer_mean))

			# Guardando log en csv
			df = pd.DataFrame(logs, columns=columnas)
			df.to_csv(self.logs_path+self.nombre+"/logs.csv", index=False)

			plt.plot(logs["loss"])
			plt.draw()
			plt.pause(0.001)
			#plt.show(block=False)

	@tf.function
	def decode(self, y_, sequence_length):
		sequence_length = tf.reshape(sequence_length, [-1])
		y_ = tf.transpose(y_, [1, 0, 2])
		decoded, _ = tf.nn.ctc_greedy_decoder(y_, sequence_length)

		return decoded[0]

	def decode_cadenas(self, decoded):
		encodedLabelStrs = [[] for i in range(20)]
		idxDict = { b : [] for b in range(20) }

		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0] # index according to [b,t]
			encodedLabelStrs[batchElement].append(label)

		cadenas = []
		for strs in encodedLabelStrs:
			cadena = self.vocabulario.decodificar(strs)
			cadenas.append(cadena)

		return cadenas

	def decode_input(self, y):
		l, nl, nf = y

		cadenas_y = []
		for i, n in enumerate(nl):
			cadena_y = self.vocabulario.decodificar(l[i, :n[0]])
			cadenas_y.append(cadena_y)

		return cadenas_y

	def WER(self, cadenas, cadenas_y):
		err_wer = []
		for i, c in enumerate(cadenas):
			cy = cadenas_y[i]
			nw = len(cy.split())
			err = wer(c.split(), cy.split())

			err_wer.append(err/nw)

			
		return err_wer

	def printDecoded(self, cadenas, cadenas_y):
		for i, c in enumerate(cadenas):
			cy = cadenas_y[i]
			print("CADENA ORIGINAL")
			print(cy)
			print("CADENA MODELO")
			print(c)

		columnas = ["original", "prediccion"]
		datos = {
			"original": cadenas_y,
			"prediccion": cadenas,
		}

		df = pd.DataFrame(datos, columns=columnas)
		df.to_csv(self.logs_path+self.nombre+"/prediccion.csv", index=False)


	#@tf.function
	def train_epoch(self, optimizer, train, epoch):
		epoch_loss = [] 
		WER = []
	
		tf.summary.trace_off()
		for x, y in train:
			y_, loss_value, grads = self.grad(x, y)
			optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

			l, nl, nf = y

			d = self.decode(y_, nf)
			cadenas = self.decode_cadenas(d)
			cadenas_y = self.decode_input(y)

			WER.append(self.WER(cadenas, cadenas_y))

			if epoch % 10 == 0:
				self.printDecoded(cadenas, cadenas_y)

			epoch_loss.append(loss_value)

		tf.summary.trace_on()

		return epoch_loss, WER

		


