from .pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from .. import ctc

"""
Se define el pipeline de entrenamiento para el dataset
"""
class DS2Pipeline(Pipeline):
    def __init__(self, model=None, features=None, vocabulario=None, 
            dataset_distrib=None):
        self.model = model
        self.features = features
        self.vocabularion = vocabulario
        self.dataset_distrib = dataset_distrib

    def fit(self, train_descrip, test_descrip, setup):
        print("[INFO] Cargando distribuciones del datasetpipeline")
        train = self.dataset_distrib(train_descrip)
        test = self.dataset_distrib(test_descrip)

        lr = setup["learning_rate"]
        bs = setup["batch_size"]
        epochs = setup["epochs"]
        i_epoch = setup["initial_epoch"]

        optimizer = Adam()

        train = train.batch(bs)
        test  = test.batch(bs)

        def loss(model, x, y, training):
            y_ = model(x, training=True)

            return get_loss(y, y_)

        def grad(model, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value = loss(model, inputs, targets, training=True)

            return loss_value, tape.gradient(loss_value, model.trainable_variables)


        train_loss_results = []
        train_accuracy_results = []

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()

            for x, y in train:
                loss_value, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_loss_avg(loss_value)

            train_loss_results.append(epoch_loss_avg.result())

            print("Epoch {:03d}: Loss: {:.3f}".format(epoch,
                    epoch_loss_avg.result()))



