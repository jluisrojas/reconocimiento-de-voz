import tensorflow as tf

class EspVocabulario:
    def __init__(self, incluir_simbolos=False):
        self.caracteres = [
            "a", "b",
            "c", "d",
            "e", "f",
            "g", "h",
            "i", "j",
            "k", "l",
            "m", "n", "ñ",
            "o", "p",
            "q", "r",
            "s", "t",
            "u", "v",
            "w", "x",
            "y", "z",
            " "
        ]

        self.vocab = {}
        for i, c in enumerate(self.caracteres):
            self.vocab[c] = i

    def __call__(self, cadena):
        res = []
        cadena = cadena.lower()
        for c in cadena:
            if c in self.vocab:
                res.append(self.vocab[c])

        return res

    def get_config():
        config = {
            "vocabulario": self.vocab
        }
        return config
