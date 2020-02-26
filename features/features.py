import abc

class ExtractorFeatures:
    @abc.abstractmethod
    def get_config(self):
        pass
