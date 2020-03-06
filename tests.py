import tensorflow as tf
from dataset.generador import GeneradorDataset
from features.spectrograma import SpectrogramaFeatures
from vocabulario.esp import EspVocabulario
from model.layers import ObtenerMask
from model.deepspeech2 import obtener_ds2

def main():
    spectrograma = SpectrogramaFeatures()
    vocabulario = EspVocabulario()
    generador = GeneradorDataset(spectrograma, vocabulario, fl=10, fs=10)
    features, labels, num_labels, num_frames = generador.generar_distribucion("dataset/common-voice/es/", "test",
        sub_ruta="clips/")
    dataset = tf.data.Dataset.from_tensor_slices((features, (labels,
        num_labels, num_frames)))

    for x, y in dataset.batch(10).take(1):
        l, n_l, n_f = y
        print(x.shape)
        print(l.shape)
        print(n_l.shape)
        print(n_f.shape)

    model = obtener_ds2(input_dim=(53, 20, 513, 1), num_convs=3)
    #model.summary()
    #print(ObtenerMask()(test))


if __name__ == "__main__":
    main()
