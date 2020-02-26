import os
from pydub import AudioSegment
import pandas as pd

"""
Metodos encargados para prepara el dataset para procesamiento
los datos deveran de tener el siguiente formato:
    - wav: para el audio
    - csv: para anotaciones de los datos
        - path
        - sentence
"""

"""
Metodo que lee un archivo mp3 y lo convierte a wav, utilizando
el mismo nombre
"""
def mp3_a_wav(src):
    audio = AudioSegment.from_mp3(src)
    audio.export(src[:-4]+".wav", format="wav")

"""
Metodo para procesar una distribucion del dataset de common voice.
Convierte el tsv a csv, y los archivos mp3 a wav y tambien su path
en el .csv
"""
def common_proc_distrib(path, tsv_file, carpeta_clips):
    df = pd.read_csv(path+tsv_file, sep="\t")
    df = df[["path", "sentence"]]

    df = df.head(20)

    num_renglones = len(df.index)

    for indice, renglon in df.iterrows():
        print("[INFO] Procesando clip {}/{}".format(indice+1,  num_renglones))
        mp3_a_wav(path+carpeta_clips+renglon["path"])

    # Convierte de mp3 a wav y cambia el path en el dataframe
    def procesar_archivo(x):
        #mp3_a_wav(path+carpeta_clips+x)
        x = x[:-4] + ".wav"
        return x

    df["path"] = df["path"].apply(procesar_archivo)

    df.to_csv(path+tsv_file[:-4]+".csv", index=False)


if __name__ == "__main__":
    mp3_a_wav("sample-dataset/common_voice_es_18306579.mp3")
