import tensorflow as tf

from src.features import SpectrogramaFeatures2
from src.vocabulario import EspVocabulario
from src.dataset import CVMDistrib, DataDescripcion


def main():
    spectrograma = SpectrogramaFeatures2(stft_fft=252)
    vocabulario = EspVocabulario()
    dataset = CVMDistrib(frame_length=0, features=spectrograma, vocabulario=vocabulario)
    train_descripcion = DataDescripcion(distribucion="train", tamano=20)
    train = dataset(train_descripcion)




if __name__ == "__main__":
    main()
