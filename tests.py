from dataset.generador import GeneradorDataset
from features.spectrograma import SpectrogramaFeatures
from model.layers import ObtenerMask

def main():
    spectrograma = SpectrogramaFeatures()
    generador = GeneradorDataset(spectrograma, fl=20, fs=20)
    test = generador.generar_distribucion("dataset/common-voice/es/", "test",
        sub_ruta="clips/")
    print(test.shape)

    print(ObtenerMask()(test))


if __name__ == "__main__":
    main()
