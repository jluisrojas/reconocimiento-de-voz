import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from .pipeline import Pipeline
#from .. import ctc
from ..ctc import get_loss

"""
Se define el pipeline de entrenamiento para Deep Speech 2, considerando:
	- CTC loss
"""
class DS2Pipeline(Pipeline):
	"""
	Constructor del pipeline de ds2
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

	def loss(self, x, y, training):
			y_ = self.model(x, training=True)

			return get_loss(y, y_)

	def grad(self, inputs, targets):
			with tf.GradientTape() as tape:
				loss_value = self.loss(inputs, targets, training=True)

			return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
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
		test = self.dataset_distrib(test_descrip)

		print(train)

		lr = setup["learning_rate"]
		bs = setup["batch_size"]
		epochs = setup["epochs"]
		i_epoch = setup["initial_epoch"]

		optimizer = Adam(lr=lr)

		train = train.batch(bs)
		test  = test.batch(bs)



		columnas = ["epoch", "loss"]
		logs = { }
		for c in columnas:
			logs[c] = []

		for epoch in range(epochs):
			print("Iniciando epoch: {}".format(epoch))
			logs["epoch"].append(epoch)

			epoch_loss = None
			
			for x, y in train:
				loss_value, grads = self.grad(x, y)
				optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

				if epoch_loss == None:
					epoch_loss = loss_value
				else:
					epoch_loss = tf.concat([epoch_loss, loss_value], 0)
				
			loss_mean = tf.reduce_mean(epoch_loss).numpy()
			logs["loss"].append(loss_mean)

			print("Epoch {:03d}: Loss: {:.3f}".format(epoch, loss_mean))

			# Guardando log en csv
			df = pd.DataFrame(logs, columns=columnas)
			df.to_csv(self.logs_path+self.nombre+"/logs.csv", index=False)



