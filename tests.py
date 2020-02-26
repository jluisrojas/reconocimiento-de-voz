from dataset.generador import GeneradorDataset
from features.spectrograma import SpectrogramaFeatures

def main():
    spectrograma = SpectrogramaFeatures()
    generador = GeneradorDataset(spectrograma)
    test = generador.generar_distribucion("dataset/common-voice/es/", "test",
        sub_ruta="clips/")

    print(test)


if __name__ == "__main__":
    main()
