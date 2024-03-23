import os
import unittest

import tensorflow as tf

from source.utils.multi_model_trainer import MultiModelTrainer
from source.utils.test_result_logger import TestResultLogger

from matplotlib import pyplot as plt


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..","..")
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/utils/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")

PLOT_SHOW = False

class TestMultiModelTrainer(unittest.TestCase):

    @classmethod
    def _get_dataset(cls):
        (x, y), _ = tf.keras.datasets.mnist.load_data()
        x = x.astype('float32') / 255
        x_subset = x[:32]
        y_subset = y[:32]
        y_subset = tf.one_hot(y_subset, 10)
        return tf.data.Dataset.from_tensor_slices((x_subset, y_subset)).batch(32)

    
    @classmethod
    def setUpClass(cls):
        print('Dataset loading...')
        cls.group_names = ['group1', 'group2']
        cls.datasets = {
            'group1': (cls._get_dataset(), cls._get_dataset()), 
            'group2': (cls._get_dataset(), cls._get_dataset())
        }
        print('Datasets loaded')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Multi Model Trainer Test")

    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(784, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        print(f"Test {self._testMethodName} completed")
        tf.keras.backend.clear_session()

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
        kwargs = {'epochs': 1, 'verbose': 0}
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

    def test_calculate_model_predictions(self):
        trainer = MultiModelTrainer(self.group_names)
        trainer.load_model(self.model)
        trainer.set_datasets(self.datasets)
        trainer.fit_all(verbose=0, epochs=10)
        test_datasets = {
            'group1': self._get_dataset(), 
            'group2': self._get_dataset()
        }
        trainer.calculate_model_predictions(test_datasets)
        self.assertEqual(len(trainer.visualizers), 2)
        self.assertEqual(len(trainer.final_results), 2)

    def test_plot_all_results(self):
        trainer = MultiModelTrainer(self.group_names)
        trainer.load_model(self.model)
        trainer.set_datasets(self.datasets)
        trainer.calculate_model_predictions({'group1': self._get_dataset(), 'group2': self._get_dataset()})
        figures = trainer.plot_all_results(n_rows=3, n_cols=3, title='All Results', fontsize=12, prediction_bar=True, show_plot=PLOT_SHOW)

        self.assertIsInstance(figures, dict)
        self.assertEqual(len(figures), len(self.group_names))
        for group in self.group_names:
            # is of type matplotlib.figure.Figure
            self.assertIsInstance(figures[group], type(plt.figure()))


if __name__ == '__main__':
    unittest.main()