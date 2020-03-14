import abc

class DataDescripcion:
    def __init__(self,
            distribucion="train",
            tamano=None):
        self.distribucion = distribucion
        self.tamano = tamano

    def get_confg(self):
        return {
            "distribucion": self.distribucion,
            "tamano": self.tamano
        }

class DataDistrib:
    @abc.abstractmethod
    def get_config(self):
        pass
