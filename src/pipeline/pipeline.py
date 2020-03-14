import abc

class Pipeline:
    @abc.abstractmethod
    def get_config(self):
        pass
