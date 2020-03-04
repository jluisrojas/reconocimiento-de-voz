from dataset.generador import GeneradorDataset
from features.spectrograma import SpectrogramaFeatures
from model.layers import ObtenerMask
from model.deepspeech2 import obtener_ds2

def main():
    spectrograma = SpectrogramaFeatures()
    generador = GeneradorDataset(spectrograma, fl=20, fs=20)
    test = generador.generar_distribucion("dataset/common-voice/es/", "test",
        sub_ruta="clips/")
    print(test.shape)

    model = obtener_ds2(input_dim=(53, 20, 513, 1), num_convs=3)
    model.summary()
    #print(ObtenerMask()(test))


if __name__ == "__main__":
    main()
