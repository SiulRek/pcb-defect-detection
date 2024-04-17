import os
import unittest

import tensorflow as tf

from source.model.helpers.image_classifiers_trainer import ImageClassifiersTrainer
from source.utils.test_result_logger import TestResultLogger

from matplotlib import pyplot as plt


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..","..")
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/model/tests/outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "ict_plots")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")

SHOW_PLOTS = False
SAVE_PLOTS = True


class TestImageClassifiersTrainer(unittest.TestCase):
    """ Tests for the ImageClassifiersTrainer class.

    The ImageClassifiersTrainer class is used to train multiple image classifiers on different datasets and evaluate their performance.
    """

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
        cls.group_names = ['group1', 'group2']
        cls.categories = [str(i) for i in range(10)]
        cls.train_datasets = {
            'group1':   cls._get_dataset(), 
            'group2':   cls._get_dataset()
        }
        cls.val_datasets = {
            'group1':   cls._get_dataset(), 
            'group2':   cls._get_dataset()
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if SAVE_PLOTS:
            os.makedirs(PLOTS_DIR, exist_ok=True)
        cls.logger = TestResultLogger(LOG_FILE)
        cls.logger.log_title("Multi Model Trainer Test")

    def setUp(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        print(f"Test {self._testMethodName} completed")
        tf.keras.backend.clear_session()

    def test_load_model(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        self.assertEqual(len(trainer.models), 2)
        self.assertEqual(trainer.models['group1'].get_config(), trainer.models['group2'].get_config())

    def test_plot_model_summary(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        fig = trainer.plot_model_summary(title='Model Summary', fontsize=12, show_plot=SHOW_PLOTS)
        self.assertIsNotNone(fig)
        if SAVE_PLOTS:
            fig.savefig(os.path.join(PLOTS_DIR, 'model_summary.png'))

    def test_set_datasets(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer._set_datasets(train_datasets=self.train_datasets, val_datasets=self.val_datasets)
        self.assertEqual(len(trainer.train_datasets), 2)
        self.assertEqual(len(trainer.val_datasets), 2)
        self.assertEqual(trainer.train_datasets['group1'], self.train_datasets['group1'])
        self.assertEqual(trainer.val_datasets['group2'], self.val_datasets['group2'])

    def test_fit_all(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0)
        self.assertEqual(len(trainer.histories), 2)
        self.assertIn('loss', trainer.histories['group1'].history)
    
    def test_fit_all_with_kwargs(self):
        kwargs = {'epochs': 1, 'verbose': 0}
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, **kwargs)
        self.assertEqual(len(trainer.histories), 2)
        self.assertIn('loss', trainer.histories['group1'].history)
    
    def test_fit_all_with_callbacks(self):
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * 0.9

        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)

        trainer.fit_all(train_datasets=self.train_datasets, 
                        val_datasets=self.val_datasets, 
                        callbacks=[lr_scheduler_callback], 
                        epochs=10, 
                        verbose=0)

        for group in self.group_names:
            history = trainer.histories[group].history
            self.assertIn('lr', history.keys())

            initial_lr = history['lr'][0]
            later_lr = history['lr'][-1]
            self.assertLess(later_lr, initial_lr)

    def test_plot_histories(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=10)
        trainer.plot_histories(plot_show=SHOW_PLOTS)
        self.assertIsNotNone(trainer.history_plot)
        if SAVE_PLOTS:
            trainer.history_plot.savefig(os.path.join(PLOTS_DIR, 'histories.png'))

    def test_calculate_model_predictions(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=10)
        test_datasets = {
            'group1': self._get_dataset(), 
            'group2': self._get_dataset()
        }
        trainer.calculate_model_predictions(test_datasets)
        self.assertEqual(len(trainer.visualizers), 2)
        self.assertEqual(len(trainer.final_results), 2)

    def test_plot_all_results(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=1)
        trainer.calculate_model_predictions(self.val_datasets)
        figures = trainer.plot_all_results(n_rows=3, n_cols=3, title='All Results', fontsize=12, 
                                           prediction_bar=True, show_plot=SHOW_PLOTS)

        self.assertIsInstance(figures, dict)
        self.assertEqual(len(figures), len(self.group_names))
        for group in self.group_names:
            self.assertIsInstance(figures[group], type(plt.figure()))

        if SAVE_PLOTS:
            for group in self.group_names:
                figures[group].savefig(os.path.join(PLOTS_DIR, f'all_results_{group}.png'))
    
    def test_plot_all_false_results(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=1)
        trainer.calculate_model_predictions(self.val_datasets)
        figures = trainer.plot_all_false_results(n_rows=3, n_cols=3, title='All False Results', 
                                                 fontsize=12, prediction_bar=True, show_plot=SHOW_PLOTS)

        self.assertIsInstance(figures, dict)
        self.assertEqual(len(figures), len(self.group_names))
        for group in self.group_names:
            self.assertIsInstance(figures[group], type(plt.figure()))
        
        if SAVE_PLOTS:
            for group in self.group_names:
                figures[group].savefig(os.path.join(PLOTS_DIR, f'all_false_results_{group}.png'))
            
    def test_plot_all_confusion_matrices(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=1)
        trainer.calculate_model_predictions(self.val_datasets)
        figures = trainer.plot_all_confusion_matrices(title='All Confusion Matrices', fontsize=12, show_plot=SHOW_PLOTS)

        self.assertIsInstance(figures, dict)
        self.assertEqual(len(figures), len(self.group_names))
        for group in self.group_names:
            self.assertIsInstance(figures[group], type(plt.figure()))
        
        if SAVE_PLOTS:
            for group in self.group_names:
                figures[group].savefig(os.path.join(PLOTS_DIR, f'all_false_results_{group}.png'))
        
    def test_calculate_evaluation_metrics(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=1)
        trainer.calculate_model_predictions(self.val_datasets)
        metrics = trainer.calculate_evaluation_metrics()
        self.assertEqual(len(metrics), 4) # + mean and stdev
        self.assertEqual(len(metrics['group1']), 4)
        self.assertEqual(len(metrics['group2']), 4)

    def test_plot_all_evaluation_metrics(self):
        trainer = ImageClassifiersTrainer(self.group_names, self.categories)
        trainer.load_model(self.model)
        trainer.fit_all(train_datasets=self.train_datasets, verbose=0, epochs=1)
        trainer.calculate_model_predictions(self.val_datasets)
        figure = trainer.plot_all_evaluation_metrics(title='All Evaluation Metrics', fontsize=12, show_plot=SHOW_PLOTS)
        
        self.assertIsInstance(figure, plt.Figure)

        if SAVE_PLOTS:
            figure.savefig(os.path.join(PLOTS_DIR, 'all_evaluation_metrics.png'))
    
    def test_plot_roc_curves(self):
        dataset = self._get_dataset()
        dataset = dataset.map(lambda x, y: (x, y[:, 0]))
        datasets = {
            'group1': dataset,
            'group2': dataset
        }
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        trainer = ImageClassifiersTrainer(self.group_names, ['0', '1'], is_multiclass=False)
        trainer.load_model(model)
        trainer.fit_all(train_datasets=datasets, verbose=0, epochs=1)
        trainer.calculate_model_predictions(datasets)
        fig = trainer.plot_roc_curves(title='ROC Curves', fontsize=12, show_plot=True)
        
        self.assertIsInstance(fig, plt.Figure)

        if SAVE_PLOTS:
            fig.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'))


if __name__ == '__main__':
    unittest.main()