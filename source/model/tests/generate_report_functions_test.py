import unittest
import os
import shutil
import pickle
from source.utils import TestResultLogger
from source.model.helpers.generate_report_functions import (
    get_experiments_data,
    sort_experiments_data,
)

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
OUTPUT_DIR = os.path.join(ROOT_DIR, "source/model/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


class TestGenerateReportFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Testing Experiment Data Retrieval and Sorting")
        cls.test_dir = os.path.join(OUTPUT_DIR, "test_experiments")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.data = {
            "experiment1": {"mean": {"accuracy": 0.9}, "std": {"accuracy": 0.1}},
            "experiment2": {"mean": {"accuracy": 0.85}, "std": {"accuracy": 0.15}},
            "experiment3": {"mean": {"accuracy": 0.95}, "std": {"accuracy": 0.05}},
        }
        for exp, exp_data in cls.data.items():
            exp_path = os.path.join(cls.test_dir, exp)
            os.makedirs(exp_path, exist_ok=True)
            with open(os.path.join(exp_path, "results.pickle"), "wb") as f:
                pickle.dump(exp_data, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        cls.logger.log_title("Completed Testing Experiment Data Retrieval and Sorting")

    def test_get_experiments_data(self):
        experiments_data = get_experiments_data(self.test_dir, "results.pickle")
        self.assertEqual(len(experiments_data), len(self.data))
        for path, data in experiments_data:
            exp_name = path.split(os.sep)[-1]
            self.assertEqual(data, self.data[exp_name])

    def test_get_experiments_data_error(self):
        with self.assertRaises(FileNotFoundError):
            get_experiments_data(self.test_dir, "results2.pickle")

    def test_sort_experiments_data(self):
        experiments_data = get_experiments_data(self.test_dir, "results.pickle")
        sorted_data = sort_experiments_data(experiments_data, ("accuracy", "mean"))
        accuracies = [exp_data[1]["mean"]["accuracy"] for exp_data in sorted_data]
        self.assertTrue(
            all(accuracies[i] >= accuracies[i + 1] for i in range(len(accuracies) - 1))
        )

    def test_sort_experiments_data_error_1(self):
        experiments_data = get_experiments_data(self.test_dir, "results.pickle")
        with self.assertRaises(ValueError):
            sort_experiments_data(experiments_data, ("accuracy", "mean2"))

    def test_sort_experiments_data_error_2(self):
        experiments_data = get_experiments_data(self.test_dir, "results.pickle")
        with self.assertRaises(ValueError):
            sort_experiments_data(experiments_data, ("accuracy2", "mean"))


if __name__ == "__main__":
    unittest.main()
