import unittest
import numpy as np
from models.production_model import ProductionModel
from model import Model  
from production_model import ProductionModel

class TestProductionModel(unittest.TestCase):
    def setUp(self):
        self.model = ProductionModel()

    def test_fit_and_generate(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        self.model.fit(X, y)
        predictions = self.model.generate(X)
        self.assertEqual(len(predictions), 3)

    def test_evaluate(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        self.model.fit(X, y)
        score = self.model.evaluate(X, y)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()
