import pytest
import numpy as np
from katabatic.models.production_model import ProductionModel
from model import Model  
from production_model import ProductionModel

@pytest.fixture
def model():
    return ProductionModel()

def test_fit_and_generate(model):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model.fit(X, y)
    predictions = model.generate(X)
    assert len(predictions) == 3

def test_evaluate(model):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model.fit(X, y)
    score = model.evaluate(X, y)
    assert isinstance(score, float)
    assert 0 <= score <= 1
