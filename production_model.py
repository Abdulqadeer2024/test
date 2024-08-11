import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .model import Model

class ProductionModel(Model):
    def __init__(self):
        super().__init__("production_model")
        self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def generate(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)
