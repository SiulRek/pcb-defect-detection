import os
import unittest

import tensorflow as tf

from source.utils.multi_model_trainer import MultiModelTrainer
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..","..")
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/utils/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")

PLOT_SHOW = False

class TestMultiModelTrainer(unittest.TestCase):

    @classmethod
    def _get_dataset(cls):
        return tf.data.Dataset.from_tensor_slices(
            (tf.random.normal((100, 100)), tf.random.normal((100,)))
        ).batch(32)
    
    @classmethod
    def setUpClass(cls):
        cls.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dense(1)
        ])
        cls.model.compile(optimizer='adam', loss='mse')

        cls.group_names = ['group1', 'group2']
        cls.datasets = {
            'group1': (cls._get_dataset(), cls._get_dataset(), cls._get_dataset()), 
            'group2': (cls._get_dataset(), cls._get_dataset(), cls._get_dataset())
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Multi Model Trainer Test")

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_load_model(self):
        trainer = MultiModelTrainer(self.group_names)
        trainer.load_model(self.model)
        self.assertEqual(len(trainer.models), 2)
        self.assertEqual(trainer.models['group1'].get_config(), trainer.models['group2'].get_config())

    def test_set_datasets(self):
        trainer = MultiModelTrainer(self.group_names)
        trainer.set_datasets(self.datasets)
        self.assertEqual(len(trainer.datasets), 2)
        self.assertEqual(trainer.datasets['group1'], self.datasets['group1'])
        self.assertEqual(trainer.datasets['group2'], self.datasets['group2'])                                               

    def test_set_datasets_error(self):
        trainer = MultiModelTrainer(self.group_names)
        with self.assertRaises(AssertionError):
            trainer.set_datasets({})

    def test_fit_all(self):
        trainer = MultiModelTrainer(self.group_names)
        trainer.load_model(self.model)
        trainer.set_datasets(self.datasets)
        trainer.fit_all(verbose=0)
        self.assertEqual(len(trainer.histories), 2)
        self.assertIn('loss', trainer.histories['group1'].history)
    
    def test_fit_all_with_kwargs(self):
        kwargs = {'epochs': 1, 'batch_size': 32}
        trainer = MultiModelTrainer(self.group_names)
        trainer.load_model(self.model)
        trainer.set_datasets(self.datasets)
        trainer.fit_all(**kwargs)
        self.assertEqual(len(trainer.histories), 2)
        self.assertIn('loss', trainer.histories['group1'].history)

    def test_plot_histories(self):
        trainer = MultiModelTrainer(self.group_names)
        trainer.load_model(self.model)
        trainer.set_datasets(self.datasets)
        trainer.fit_all(verbose=0, epochs=10)
        trainer.plot_histories(plot_show=PLOT_SHOW)
        self.assertIsNotNone(trainer.history_plot)

if __name__ == '__main__':
    unittest.main()