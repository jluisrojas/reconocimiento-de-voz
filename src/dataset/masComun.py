import pandas as pd
import numpy as np

def main():
	df = pd.read_csv("./common-voice/es/train.csv")
	vocabulario = {}
	palabra = "encuentra" #1070
	audios = []
	cadenas = []
	for indice, renglon in df.iterrows():
		guardar = False
		cadena = renglon["sentence"]
		for palabra in cadena.split():
			if palabra in vocabulario:
				vocabulario[palabra] += 1
			else:
				vocabulario[palabra] = 1

			if palabra == "encuentra":
				guardar = True

		if guardar:
			audios.append(renglon["path"])
			cadenas.append(cadena)
			

	vocabulario = {k: v for k,v in sorted(vocabulario.items(), key=lambda item: item[1])}
	#print(vocabulario)

	particion = 856
	columnas = ["path", "sentence"]
	datosTrain = {
		"path": audios[:particion],
		"sentence": cadenas[:particion]
	}
	datosTest= {
		"path": audios[particion:],
		"sentence": cadenas[particion:]
	}

	dfTrain = pd.DataFrame(datosTrain, columns=columnas)
	dfTrain.to_csv("./common-voice/es/trainEncuentra.csv", index=False)
	print(dfTrain)

	dfTest = pd.DataFrame(datosTest, columns=columnas)
	dfTest.to_csv("./common-voice/es/testEncuentra.csv", index=False)
	print(dfTest)


if __name__ == "__main__":
	main()
