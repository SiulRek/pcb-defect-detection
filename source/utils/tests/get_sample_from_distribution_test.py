import os

import unittest
from source.utils import get_sample_from_distribution  
from source.utils import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..","..")
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/utils/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")

        
class TestGetSampleFromDistribution(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Get Sample From Distribution Test")

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_gaussian_distribution(self):
        data = {"distribution": "gaussian", "loc": 0, "scale": 1}
        sample = get_sample_from_distribution(data)
        self.assertIsInstance(sample, float)
        
    def test_gaussian(self):
        data = {"distribution": "gaussian", "loc": 0, "scale": 1}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_uniform(self):
        data = {"distribution": "uniform", "low": 1, "high": 2}
        sample = get_sample_from_distribution(data)
        self.assertIsInstance(sample, float)
        self.assertTrue(1 <= sample <= 2)

    def test_exponential(self):
        data = {"distribution": "exponential", "scale": 1}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_poisson(self):
        data = {"distribution": "poisson", "lam": 3}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_binomial(self):
        data = {"distribution": "binomial", "n": 10, "p": 0.5}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_gamma(self):
        data = {"distribution": "gamma", "shape": 2, "scale": 1}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_beta(self):
        data = {"distribution": "beta", "a": 0.5, "b": 0.5}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_lognormal(self):
        data = {"distribution": "lognormal", "mean": 0, "sigma": 1}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_laplace(self):
        data = {"distribution": "laplace", "loc": 0, "scale": 1}
        self.assertIsInstance(get_sample_from_distribution(data), float)

    def test_invalid_distribution(self):
        data = {"distribution": "unknown", "param": 1}
        with self.assertRaises(ValueError):
            get_sample_from_distribution(data)

    def test_invalid_distribution_arguments(self):
        data = {"distribution": "gaussian", "invalid_argument": 1}
        with self.assertRaises(ValueError):
            get_sample_from_distribution(data)

    def test_missing_distribution_key(self):
        data = {"mean": 0, "std_dev": 1}
        with self.assertRaises(KeyError):
            get_sample_from_distribution(data)


if __name__ == "__main__":
    unittest.main()
