import unittest
from python_code.utils.get_sample_from_distribution import get_sample_from_distribution  

class TestGetSampleFromDistribution(unittest.TestCase):

    def test_gaussian_distribution(self):
        data = {'distribution': 'gaussian', 'loc': 0, 'scale': 1}
        sample = get_sample_from_distribution(data)
        self.assertIsInstance(sample, float)

    def test_uniform_distribution(self):
        data = {'distribution': 'uniform', 'low': 1, 'high': 2}
        sample = get_sample_from_distribution(data)
        self.assertIsInstance(sample, float)
        self.assertTrue(1 <= sample <= 2)

    def test_invalid_distribution(self):
        data = {'distribution': 'unknown', 'param': 1}
        with self.assertRaises(ValueError):
            get_sample_from_distribution(data)

    def test_invalid_distribution_arguments(self):
        data = {'distribution': 'gaussian', 'invalid_argument': 1}
        with self.assertRaises(ValueError):
            get_sample_from_distribution(data)

    def test_missing_distribution_key(self):
        data = {'mean': 0, 'std_dev': 1}
        with self.assertRaises(KeyError):
            get_sample_from_distribution(data)


if __name__ == '__main__':
    unittest.main()
