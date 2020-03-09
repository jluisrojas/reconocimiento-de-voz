import tensorflow as tf
from dataset.generador import GeneradorDataset
from features.spectrograma import SpectrogramaFeatures
from vocabulario.esp import EspVocabulario
from model.layers import ObtenerMask
from model.deepspeech2 import obtener_ds2
from ctc import ctc_loss
from tensorflow.keras.optimizers import Adam

def main():
    spectrograma = SpectrogramaFeatures()
    vocabulario = EspVocabulario()
    generador = GeneradorDataset(spectrograma, vocabulario, fl=10, fs=10)
    features, labels, num_labels, num_frames = generador.generar_distribucion("dataset/common-voice/es/", "test",
        sub_ruta="clips/")
    dataset = tf.data.Dataset.from_tensor_slices((features, (labels,
        num_labels, num_frames)))

    def resize(x, y):
        x = tf.image.resize(x, [20, 20])
        return x, y

    #dataset = dataset.map(resize)

    model = obtener_ds2(input_dim=(105, 10, 513, 1), num_convs=3,
            num_labels=len(vocabulario.caracteres))

    for x, y in dataset.batch(10).take(1):
        l, n_l, n_f = y
        print(x.shape)
        print(l.shape)
        print(n_l.shape)
        print(n_f.shape)

        #print(model(x))

    model.summary()

    def loss(model, x, y, training):
        y_ = model(x, training=True)

        return ctc_loss.get_loss(y, y_)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = Adam()

    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 1

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()

        for x, y in dataset.batch(10).take(1):
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)

        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(epoch,
                epoch_loss_avg.result()))




if __name__ == "__main__":
    main()
