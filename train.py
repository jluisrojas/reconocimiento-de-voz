from src.features import SpectrogramaFeatures
from src.vocabulario import EspVocabulario
from src.model import obtener_ds2

from src.dataset import CVMDistrib, DataDescripcion 
from src.pipeline import DS2Pipeline

# Prueba desde ubuntu
def main():
    print("[INFO] Inicializando modulos para pipeline")
	spectrograma = SpectrogramaFeatures(stft_fft=252)
	vocabulario = EspVocabulario()

	print("[INFO] Cargando modelo Deep Speech 2")
	model = obtener_ds2(input_dim=(987, 127, 1), num_convs=1,
		num_labels=len(vocabulario.caracteres)+1)

	model.summary()

	dataset = CVMDistrib(frame_length=0, features=spectrograma, vocabulario=vocabulario)

	pipeline = DS2Pipeline(model=model, features=spectrograma,
		vocabulario=vocabulario, dataset_distrib=dataset)

	print(pipeline.get_config())

	train_descripcion = DataDescripcion(distribucion="train", tamano=50)
	test_descripcion = DataDescripcion(distribucion="test", tamano=50)


	print("[INFO] Entrenando modelo")
	pipeline.fit(train_descripcion, test_descripcion,
			{
				"learning_rate": 1e-3,
				"batch_size": 50,
				"epochs": 200,
				"initial_epoch": 0
			}
		)

if __name__ == "__main__":
	main()
