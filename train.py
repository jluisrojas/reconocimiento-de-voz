from src.features import SpectrogramaFeatures
from src.vocabulario import EspVocabulario
from src.model import obtener_ds2

from src.dataset import CVMDistrib, DataDistrib
from src.pipeline import DS2Pipeline

def main():
    print("[INFO] Inicializando modulos para pipeline")
    spectrograma = SpectrogramaFeatures()
    vocabulario = EspVocabulario()

    print("[INFO] Cargando modelo Deep Speech 2")
    model = obtener_ds2(input_dim=(None, 10, 513, 1), num_convs=3,
            num_labels=len(vocabulario.caracteres))

    dataset = CVMDistrib(spectrograma, vocabulario)

    pipeline = DS2Pipeline(model=model, features=spectrograma,
            vocabulario=vocabulario, dataset_distrib=dataset)

    train_descripcion = DataDistrib(distribucion="test", tamano=5)
    test_descripcion = DataDistrib(distribucion="test", tamano=5)


    print("[INFO] Entrenando modelo")
    pipeline.fit(train_descripcion, test_descripcion,
            {
                "learning_rate": 1e-6,
                "batch_size": 5,
                "epochs": 2,
                "initial_epoch": 0
            }
        )

if __name__ == "__main__":
    main()
