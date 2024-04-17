from copy import deepcopy
import io

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf

from source.model.helpers.image_classifier_visualizer import ImageClassifierVisualizer


class ImageClassifiersTrainer():
    """ 
    A class to train multiple image classifiers of the same type on different datasets
    and visualize their results.
    """
    def __init__(self, group_names, category_names, is_multiclass=True):
        """
        Initializes the ImageClassifiersTrainer.

        Args:
        - group_names (list): Names of the groups of classifiers.
        - category_names (list): Names of the categories.
        """
        self.group_names = group_names
        self.groups_num = len(group_names)
        self.models = {}
        self.is_multiclass = is_multiclass
        self.histories = {}
        self.train_datasets = None
        self.val_datasets = None
        self.model_predictions_calculated = False
        self.final_results = {group: {} for group in group_names}
        self.visualizers = {}
        self.visualizers = {group: ImageClassifierVisualizer(category_names, is_multiclass) for group in group_names}

    def load_model(self, model):
        """ 
        Load a model and create copies for each group.

        Args:
        - model: A compiled Keras model.
        """
        compile_config = model.get_compile_config()
        self.models = {group: tf.keras.models.clone_model(model) for group in self.group_names}
        for m in self.models.values():
            m.compile(**compile_config)  

    def plot_model_summary(self, title='Model Summary', fontsize=10, show_plot=True):
        """ 
        Plot the summary of each trained model.
        
        Args:
        - fontsize (int, optional): Font size for text in the plot.
        - show_plot (bool, optional): Whether to display the plot.
        
        Returns:
        - fig: Matplotlib figure for the model configuration.
        """
        model = list(self.models.values())[0]

        summary_str = io.StringIO()
        model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        summary_text = summary_str.getvalue()
        summary_str.close()

        fig, ax = plt.subplots(figsize=(12, len(summary_text.split('\n')) * 0.4)) 
        ax.axis('off')
        ax.text(0.01, 0.99, title, fontsize=fontsize + 2, fontweight='bold', verticalalignment='top', transform=ax.transAxes)
        ax.text(0.01, 0.94, summary_text, fontsize=fontsize, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
        plt.subplots_adjust(top=0.85) 
        if show_plot:
            plt.show()

        return fig

    def _set_datasets(self, train_datasets, val_datasets):      
        assert set(train_datasets.keys()) == set(self.group_names), "Train Datasets must be the same as group names"
        self.train_datasets = train_datasets

        if val_datasets is not None:
            assert set(val_datasets.keys()) == set(self.group_names), "Datasets must be the same as group names"
            self.val_datasets = val_datasets
        
    def fit_all(self, train_datasets, val_datasets=None, callbacks=None, **kwargs):
        """ 
        Fit all models on their respective datasets.

        Args:
        - datasets (dict): A dictionary with keys as group names and values as tuples of training and validation datasets.
        - callbacks (list, optional): List of callbacks to pass to the fit method.
        - kwargs: Additional arguments to pass to the fit method.
        """
        self._set_datasets(train_datasets, val_datasets)

        for group in self.group_names:
            fit_params = {}
            if callbacks is not None:
                fit_params['callbacks'] = deepcopy(callbacks)
            if val_datasets is not None:
                fit_params['validation_data'] = self.val_datasets[group]
            self.histories[group] = self.models[group].fit(self.train_datasets[group], **fit_params, **kwargs)
            final_results = {}
            for metric, values in self.histories[group].history.items():
                final_results[metric] = values[-1]
            self.final_results[group] = final_results

        for group, results in self.final_results.items():
            print(f"Final results for group {group}: {results}")
        
    def plot_histories(self, metrics=None, plot_show=True):
        """ 
        Plot the training history of all models.

        Args:
        - metrics (list, optional): List of metrics to plot. If None, all metrics are plotted.
        - plot_show (bool, optional): Whether to display the plot.

        Returns:
        - fig: Matplotlib figure for the training history.
        """
        if not metrics:
            first_history = next(iter(self.histories.values()))
            metrics = list(first_history.history.keys())

        n_cols = 1
        n_rows = len(metrics)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

        for i, metric in enumerate(metrics):
            ax = axes[i] if n_rows > 1 else axes 

            ax.set_title(f'Training {metric}')
            for group, history in self.histories.items():
                values = history.history[metric]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=f'{group}')

            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()

        fig.tight_layout()
        if plot_show:
            plt.show()
        self.history_plot = fig  
    
    def calculate_model_predictions(self, test_datasets):
        """
        Calculate the predictions of all models on the test datasets.

        Args:
        - test_datasets (dict): A dictionary with keys as group names and values as test datasets.
        """
        assert set(test_datasets.keys()) == set(self.group_names), "Datasets must be the same as group names"
        if self.visualizers == {}:
            raise Exception("Visualizers have not been initialized. Please set initialize_classifier_visualizers=True in the constructor")
        for group, model in self.models.items():
            self.visualizers[group].calculate_model_predictions(model, test_datasets[group])
        self.model_predictions_calculated = True

    def plot_all_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        """ 
        Plot the results of all models.

        Args:
        - n_rows (int, optional): Number of rows in the plot.
        - n_cols (int, optional): Number of columns in the plot.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size for text in the plot.
        - prediction_bar (bool, optional): Whether to show the prediction bar.
        - show_plot (bool, optional): Whether to display the plot.

        Returns:
        - figures (dict): A dictionary with group names as keys and Matplotlib figures as values.
        """
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first")
        
        title = '' if title is None else title
        figures = {}
        for group, visualizer in self.visualizers.items():
            group_title = f"{title} - {group}"
            figures[group] = visualizer.plot_results(n_rows, n_cols, group_title, fontsize, prediction_bar, show_plot)
        return figures
    
    def plot_all_false_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        """ 
        Plot the false results of all models.

        Args:
        - n_rows (int, optional): Number of rows in the plot.
        - n_cols (int, optional): Number of columns in the plot.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size for text in the plot.
        - prediction_bar (bool, optional): Whether to show the prediction bar.
        - show_plot (bool, optional): Whether to display the plot.

        Returns:
        - figures (dict): A dictionary with group names as keys and Matplotlib figures as values.
        """
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first")
        
        title = '' if title is None else title
        figures = {}
        for group, visualizer in self.visualizers.items():
            group_title = f"{title} - {group}"
            figures[group] = visualizer.plot_false_results(n_rows, n_cols, group_title, fontsize, prediction_bar, show_plot)
        return figures
    
    def plot_all_confusion_matrices(self, title=None, fontsize=12, fig_size=(10, 10), show_plot=True):
        """ 
        Plot the confusion matrices of all models.

        Args:
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size for text in the plot.
        - show_plot (bool, optional): Whether to display the plot.

        Returns:
        - figures (dict): A dictionary with group names as keys and Matplotlib figures as values.
        """
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first")

        title = '' if title is None else title    
        figures = {}
        for group, visualizer in self.visualizers.items():
            group_title = f"{title} - {group}"
            figures[group] = visualizer.plot_confusion_matrix(group_title, fontsize=fontsize, fig_size=fig_size, show_plot=show_plot)
        return figures
    
    def plot_all_evaluation_metrics(self, average='macro', title=None, fontsize=12, fig_size=(5, 3), show_plot=True):
        """
        Plot the evaluation metrics of all models.
        
        Args:
        - average (str, optional): Type of averaging used for multiclass classification. Default is 'macro'.
        - title (str, optional): Title of the plot. If None, no title is set. Default is None.
        - fontsize (int, optional): Font size for text in the plot. Default is 12.
        - show_plot (bool, optional): Whether to display the plot. Default is True.
        
        Returns:
        - figure: Matplotlib figure for the evaluation metrics.
        """
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first.")

        title = title if title is not None else '' 

        metrics = {group: visualizer.calculate_evaluation_metrics(average=average) for group, visualizer in self.visualizers.items()}

        first_group_metrics = next(iter(metrics.values()))
        metrics['average'] = {}
        metrics['std'] = {}

        for metric_name, _ in first_group_metrics.items():
            metric_values = [metrics[group][metric_name] for group in self.group_names]
            average_value = sum(metric_values) / len(metric_values)
            metrics['average'][metric_name] = average_value
            metrics['std'][metric_name] = sum([(value - average_value) ** 2 for value in metric_values]) / len(metric_values)

        metrics_header = '        ' + ',   '.join(first_group_metrics.keys()) + '\n\n'
        metrics_lines = [f"{group}:   " + ',  '.join(f"{value:.2f}" for value in group_metrics.values()) for group, group_metrics in metrics.items()]
        metrics_text = metrics_header + '\n'.join(metrics_lines)

        figure = ImageClassifierVisualizer(None).plot_text(text=metrics_text,
                                                        title=title,
                                                        fontsize=fontsize,
                                                        show_plot=show_plot,
                                                        fig_size=fig_size)
        return figure

    def plot_roc_curves(self, title='ROC Curves for All Models', fontsize=12, show_plot=True):
        """
        Plot the ROC curves of all models on one plot and return the figure.

        Args:
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size for text in the plot.
        - show_plot (bool, optional): Whether to display the plot.

        Returns:
        - fig: Matplotlib figure object containing the ROC curves.
        """
        if not self.model_predictions_calculated:
            raise Exception("Model predictions have not been calculated. Please run calculate_model_predictions first.")
        
        if self.is_multiclass:
            raise Exception("ROC curves are not available for multiclass classification.")

        fig, ax = plt.subplots(figsize=(10, 8))
        
        for group, visualizer in self.visualizers.items():
            y_true = [label[1] for label in visualizer.np_dataset]
            probs = visualizer.predictions 
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, lw=2, label=f'ROC curve for {group} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=fontsize)
        ax.set_ylabel('True Positive Rate', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.legend(loc="lower right")

        if show_plot:
            plt.show()

        return fig
    


    

    
    