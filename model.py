from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def generate(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass
