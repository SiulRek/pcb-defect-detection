import tensorflow as tf
import matplotlib.pyplot as plt

from source.utils.image_classifier_visualizer import ImageClassifierVisualizer


class MultiModelTrainer():
    def __init__(self, group_names, initialize_classifier_visualizers=True):
        self.group_names = group_names
        self.groups_nums = len(group_names)
        self.models = {}
        self.histories = {}
        self.model_predictions_calculated = False
        self.final_results = {group: {} for group in group_names}
        self.visualizers = {}
        if initialize_classifier_visualizers:
            self.visualizers = {group: ImageClassifierVisualizer(group_names) for group in group_names}

    def load_model(self, model):
        optimizer_config = model.optimizer.get_config() 
        self.models = {group: tf.keras.models.clone_model(model) for group in self.group_names}
        for m in self.models.values():
            new_optimizer = type(model.optimizer).from_config(optimizer_config) 
            m.compile(optimizer=new_optimizer, loss=model.loss, metrics=model.metrics)  

    def set_datasets(self, datasets):
        assert set(datasets.keys()) == set(self.group_names), "Datasets must be the same as group names"
        self.datasets = datasets
    
    def fit_all(self, **args):
        for group in self.group_names:
            self.histories[group] = self.models[group].fit(self.datasets[group][0], 
                                                           validation_data=self.datasets[group][1], **args)
            final_results = {}
            for metric, values in self.histories[group].history.items():
                final_results[metric] = values[-1]
            self.final_results[group] = final_results

        for group, results in self.final_results.items():
            print(f"Final results for group {group}: {results}")
        
    def plot_histories(self, metrics=None, plot_show=True):
            if not metrics:
                first_history = next(iter(self.histories.values()))
                metrics = list(first_history.history.keys())

            n_cols = 1
            n_rows = len(metrics)

            plt.figure(figsize=(12, 4 * n_rows))

            for i, metric in enumerate(metrics):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.title(f'Training {metric}')

                for group, history in self.histories.items():
                    values = history.history[metric]
                    epochs = range(1, len(values) + 1)
                    plt.plot(epochs, values, label=f'{group}')

                plt.xlabel('Epochs')
                plt.ylabel(metric.capitalize())
                plt.legend()

            plt.tight_layout()
            if plot_show:
                plt.show()

            self.history_plot = plt.gcf()
    
    def calculate_model_predictions(self, test_datasets):
        assert set(test_datasets.keys()) == set(self.group_names), "Datasets must be the same as group names"
        if self.visualizers == {}:
            raise Exception("Visualizers have not been initialized. Please set initialize_classifier_visualizers=True in the constructor")
        for group, model in self.models.items():
            self.visualizers[group].calculate_model_predictions(model, test_datasets[group])
        self.model_predictions_calculated = True

    # ChatGPT please make tests for the following three methods. Many thanks.

    def plot_all_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first")
        
        figures = {}
        for group, visualizer in self.visualizers.items():
            figures[group] = visualizer.plot_results(n_rows, n_cols, title, fontsize, prediction_bar, show_plot)
        return figures
    
    def plot_all_false_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first")
        
        figures = {}
        for group, visualizer in self.visualizers.items():
            figures[group] = visualizer.plot_false_results(n_rows, n_cols, title, fontsize, prediction_bar, show_plot)
        return figures
    
    def plot_all_confusion_matrices(self, title=None, fontsize=12, show_plot=True):
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first")
        
        figures = {}
        for group, visualizer in self.visualizers.items():
            figures[group] = visualizer.plot_confusion_matrix(title, fontsize, show_plot)
        return figures
    


    

    
    