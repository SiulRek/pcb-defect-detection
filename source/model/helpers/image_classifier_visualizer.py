import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

import seaborn as sns
from sklearn.metrics import confusion_matrix


class ImageClassifierVisualizer:
    """ Class for visualizing image classification results and model predictions."""
    def __init__(self, class_names, is_multiclass=True):
        """ Initializes the ImageClassifierVisualizer class.

        Args:
        - class_names (list): List of class names.
        - is_multiclass (bool, optional): Whether the classification task is multiclass or binary.
        """
        self.class_names = class_names
        self.num_classes = None
        if class_names is not None:
            self.num_classes = len(class_names)
        self.model_predictions_prepared = False
        self.is_multiclass = is_multiclass

    def _unbatch_dataset(self, dataset):
        try:
            return dataset.unbatch()
        except ValueError:
            return dataset

    def _prepare_plot(self, fig, title=None, fontsize=12, show_plot=True):  
        if title:
            fig.suptitle(title, fontsize=fontsize)
        fig.tight_layout()  
        if show_plot:
            plt.show() 
        
        return fig
    
    def _prepare_dataset(self, dataset, shuffle=True):
        if isinstance(dataset, np.ndarray):
            return dataset
        unbatched_dataset = self._unbatch_dataset(dataset)
        preprocessed_dataset = np.array(list(unbatched_dataset.as_numpy_iterator()), dtype=object)
        if shuffle:
            np.random.shuffle(preprocessed_dataset)
        return preprocessed_dataset

    def _filter_np_dataset(self, np_dataset, classes):
        class_indices = [self.class_names.index(c) for c in classes]
        if self.is_multiclass:
            boolean_mask = np.array([np.argmax(label) in class_indices for _, label in np_dataset])
        else:
            boolean_mask = np.array([label in class_indices for _, label in np_dataset])    
        filtered_data = np_dataset[boolean_mask]
        return filtered_data, boolean_mask
    
    def _shuffle_in_unison(self, np_dataset1, np_dataset2):
        assert len(np_dataset1) == len(np_dataset2), "Datasets must be the same length"
        
        p = np.random.permutation(len(np_dataset1))
        return np_dataset1[p], np_dataset2[p]
    
    def _get_label_name(self, label):
        if self.is_multiclass:
            return self.class_names[np.argmax(label)]
        else:
            return self.class_names[label]

    def plot_images(self, combined_dataset, n_rows=1, n_cols=1, title=None, 
                    fontsize=12, show_plot=True):
        """ Plot images from a TensorFlow dataset in a grid.

        Args:
        - combined_dataset (tf.data.Dataset): TensorFlow dataset consisting of tuples (image, label).
        - n_rows (int, optional): Number of rows in the grid.
        - n_cols (int, optional): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.

        Returns:
        - fig: Figure of the images.
        """
        np_dataset = self._prepare_dataset(combined_dataset)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes = np.atleast_2d(axes).flatten()
        for i, ax in enumerate(axes):
            if i >= n_rows * n_cols:
                break

            image, label = np_dataset[i]
            if image.shape[-1] == 1:
                ax.imshow(image[:, :, 0], cmap='gray')
            else:
                ax.imshow(image.astype("uint8"))
            ax.axis('off')

            if label is not None:
                label_name = self._get_label_name(label)
                ax.set_title(label_name, fontsize=fontsize)

        return self._prepare_plot(fig, title, fontsize, show_plot)
    
    @tf.autograph.experimental.do_not_convert
    def plot_class_specific_images(self, combined_dataset, classes, n_rows=1, n_cols=1,
                                    title=None, fontsize=12, show_plot=True):
        """ Plot images from a TensorFlow dataset in a grid.

        Args:
        - combined_dataset (tf.data.Dataset): TensorFlow dataset consisting of tuples (image, label).
        - classes (list): List of classes to plot images for.
        - n_rows (int, optional): Number of rows in the grid.
        - n_cols (int, optional): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.

        Returns:
        - fig: Figure of the images.	
        """
        np_dataset = self._prepare_dataset(combined_dataset)
        filtered_dataset, _ = self._filter_np_dataset(np_dataset, classes)
        fig = self.plot_images(filtered_dataset, n_rows=n_rows, n_cols=n_cols, title=title, fontsize=fontsize, show_plot=show_plot)
        return fig
    
    def plot_image_comparisons(self, combined_dataset_1, combined_dataset_2, n_rows=1,
                                n_cols=1, title=None, fontsize=12, show_plot=True):
        """ Plot images from two TensorFlow datasets side by side in a grid.

        Args:
        - combined_dataset_1 (tf.data.Dataset): First TensorFlow dataset consisting of tuples (image, label).
        - combined_dataset_2 (tf.data.Dataset): Second TensorFlow dataset consisting of tuples (image, label).
        - n_rows (int, optional): Number of rows in the grid.
        - n_cols (int, optional): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.

        Returns:
        - fig: Figure of the images.
        """
        np_dataset_1 = self._prepare_dataset(combined_dataset_1, shuffle=False)
        np_dataset_2 = self._prepare_dataset(combined_dataset_2, shuffle=False)
        np_dataset_1, np_dataset_2 = self._shuffle_in_unison(np_dataset_1, np_dataset_2)

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3))
        axes = np.atleast_2d(axes).flatten()
        for i, ax in enumerate(axes):
            if i >= n_rows * n_cols * 2:
                break

            image, label = np_dataset_2[i-1] if i % 2 else np_dataset_1[i]
            if image.shape[-1] == 1:
                ax.imshow(image[:, :, 0], cmap='gray')
            else:
                ax.imshow(image.astype("uint8"))
            ax.axis('off')
            
            subtitle = f'{i//2}. '
            if label is not None:
                subtitle += self._get_label_name(label)
            ax.set_title(subtitle, fontsize=fontsize)

        return self._prepare_plot(fig, title, fontsize, show_plot)  


    def plot_class_specific_image_comparisons(self, combined_dataset_1, combined_dataset_2, classes,
                                 n_rows=1, n_cols=1, title=None, fontsize=12, show_plot=True):
        """ Plot images from two TensorFlow datasets side by side in a grid.

        Args:
        - combined_dataset_1 (tf.data.Dataset): First TensorFlow dataset consisting of tuples (image, label).
        - combined_dataset_2 (tf.data.Dataset): Second TensorFlow dataset consisting of tuples (image, label).
        - classes (list): List of classes to plot images for.
        - n_rows (int, optional): Number of rows in the grid.
        - n_cols (int, optional): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.

        Returns:
        - fig: Figure of the images.
        """
        np_dataset_1 = self._prepare_dataset(combined_dataset_1, shuffle=False)
        np_dataset_2 = self._prepare_dataset(combined_dataset_2, shuffle=False)
        filtered_dataset_1, _ = self._filter_np_dataset(np_dataset_1, classes)
        filtered_dataset_2, _ = self._filter_np_dataset(np_dataset_2, classes)
        filtered_dataset_1, filtered_dataset_2 = self._shuffle_in_unison(filtered_dataset_1, filtered_dataset_2)
        fig = self.plot_image_comparisons(filtered_dataset_1, filtered_dataset_2, n_rows=n_rows, n_cols=n_cols, title=title, fontsize=fontsize, show_plot=show_plot)
        return fig

    def calculate_model_predictions(self, model, dataset):
        """ Calculate model predictions and dataset, call this to visualize results with according methods.

        Args:
        - model (tf.keras.Model): Trained TensorFlow model.
        - dataset (tf.data.Dataset): TensorFlow dataset consisting of tuples (image, label).

        Returns:
        - fig: Figure of the images.
        """
        self.model = model
        self.predictions = model.predict(dataset)
        self.np_dataset = self._prepare_dataset(dataset, shuffle=True)
        self.model_predictions_prepared = True

    def _plot_results(self, np_dataset, predictions, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        fig, axes = plt.subplots(n_rows, n_cols * (2 if prediction_bar else 1), figsize=(n_cols * (6 if prediction_bar else 3), n_rows * 4)) 
        axes = np.atleast_2d(axes).flatten()

        for i in range(n_rows * n_cols):
            img_ax = axes[i * (2 if prediction_bar else 1)]
            image, true_label = np_dataset[i]
            predicted_probs = predictions[i]

            if self.is_multiclass:
                true_label_index = np.argmax(true_label)
                predicted_label_index = np.argmax(predicted_probs)
            else:
                predicted_label_index = int(np.round(predicted_probs, 2))
                true_label_index = true_label

            img_ax.imshow(image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
            img_ax.axis('off')

            title_color = 'green' if predicted_label_index == true_label_index else 'red'
            range_len = len(self.class_names) if isinstance(true_label, np.ndarray) else 2
            bar_colors = ['green' if j == true_label_index else 'blue' for j in range(range_len)]
            if predicted_label_index != true_label_index:
                bar_colors[predicted_label_index] = 'red'

            true_label_name = self.class_names[true_label_index]
            predicted_label_name = self.class_names[predicted_label_index]
            img_ax.set_title(f'Predicted: {predicted_label_name}\nTrue: {true_label_name}', fontsize=fontsize, color=title_color)

            if prediction_bar:
                bar_ax = axes[i * 2 + 1]
                if self.is_multiclass:
                    bar_ax.bar(range(len(predicted_probs)), predicted_probs, color=bar_colors)
                else:
                    concat_probs = np.concatenate([1 - predicted_probs, predicted_probs])
                    bar_ax.bar(['0', '1'], concat_probs, color=bar_colors)
                bar_ax.set_xticks(range(len(self.class_names)))
                bar_ax.set_xticklabels(self.class_names, rotation=90)
                bar_ax.set_ylim(0, 1)

        return self._prepare_plot(fig, title, fontsize, show_plot) 


    def plot_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        """ Plots a set of images along with their predicted and true labels, with an optional prediction bar.

        Args:
        - n_rows (int): Number of rows in the grid.
        - n_cols (int): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.
        - prediction_bar (bool): Whether to add a prediction bar.
        - show_plot (bool): Whether to display the plot.

        Returns:
        - fig: Figure of the images.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")
        
        return self._plot_results(self.np_dataset, self.predictions, n_rows=n_rows, n_cols=n_cols, title=title, 
                           fontsize=fontsize, prediction_bar=prediction_bar, show_plot=show_plot)

    def plot_class_specific_results(self, classes, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        """ Plots a set of images along with their predicted and true labels, with an optional prediction bar.

        Args:
        - classes (list): List of classes to plot images for.
        - n_rows (int): Number of rows in the grid.
        - n_cols (int): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.
        - prediction_bar (bool): Whether to add a prediction bar.
        - show_plot (bool): Whether to display the plot.

        Returns:
        - fig: Figure of the images.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")
        
        filtered_dataset, boolean_mask = self._filter_np_dataset(self.np_dataset, classes)
        filtered_predictions = self.predictions[boolean_mask]
        return self._plot_results(filtered_dataset, filtered_predictions, n_rows=n_rows, n_cols=n_cols, title=title, 
                           fontsize=fontsize, prediction_bar=prediction_bar, show_plot=show_plot)
    
    def plot_false_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        """ Plots a set of images along with their predicted and true labels, with an optional prediction bar.

        Args:
        - n_rows (int): Number of rows in the grid.
        - n_cols (int): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.
        - prediction_bar (bool): Whether to add a prediction bar.
        - show_plot (bool): Whether to display the plot.

        Returns:
        - fig: Figure of the images.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")
        if self.is_multiclass:
            boolean_mask = np.argmax(self.predictions, axis=1) != np.argmax([label for _, label in self.np_dataset], axis=1)
        else:
            boolean_mask = np.round(self.predictions.flatten()) != [label for _, label in self.np_dataset]
        filtered_dataset = self.np_dataset[boolean_mask]
        filtered_predictions = self.predictions[boolean_mask]
        return self._plot_results(filtered_dataset, filtered_predictions, n_rows=n_rows, n_cols=n_cols, title=title, 
                           fontsize=fontsize, prediction_bar=prediction_bar, show_plot=show_plot)
    
    def _get_label_indices(self, labels):
        if self.is_multiclass:
            return np.argmax(labels, axis=1)
        else:
            return np.round(labels).astype(int)
        
    def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, fontsize=12, 
                                fig_size=(10, 10), show_plot=True):
        """ 
        Plot the confusion matrix of the model.

        Args:
        - normalize (bool, optional): Whether to normalize the confusion matrix.
        - title (str, optional): Title of the plot.
        - cmap (matplotlib.colors.Colormap, optional): Colormap to use for the plot.
        - fontsize (int, optional): Font size for text in the plot.
        - fig_size (tuple, optional): Size of the figure.
        - show_plot (bool, optional): Whether to display the plot.

        Returns:	
        - fig: Figure of the confusion matrix.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")

        true_indices = self._get_label_indices([label for _, label in self.np_dataset])
        predicted_indices = self._get_label_indices(self.predictions)
        cm = confusion_matrix(true_indices, predicted_indices)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=fig_size)
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_ylabel('True label', fontsize=fontsize)
        ax.set_xlabel('Predicted label', fontsize=fontsize)

        return self._prepare_plot(fig, title, fontsize, show_plot)

    def calculate_evaluation_metrics(self, average='macro'):
        """
        Calculate the evaluation metrics for the model using true labels and predictions.

        Args:
        - average (str, optional): The type of averaging to perform on the data.

        Returns:
        - dict: A dictionary containing the computed metrics.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call calculate_model_predictions first.")

        true_index = self._get_label_indices([label for _, label in self.np_dataset])
        predicted_index = self._get_label_indices(self.predictions)

        accuracy = accuracy_score(true_index, predicted_index)
        precision = precision_score(true_index, predicted_index, average=average)
        recall = recall_score(true_index, predicted_index, average=average)
        f1 = f1_score(true_index, predicted_index, average=average)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_text(self, text, title='Evaluation Metrics', fontsize=12, fig_size=(1.5, 2), show_plot=True):
        """
        Create a plot with the provided text displayed.

        Args:
        - text (str): Text to display in the plot.
        - title (str, optional): Title of the plot. Defaults to 'Evaluation Metrics'.
        - fontsize (int, optional): Font size for text in the plot. Defaults to 12.
        - show_plot (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
        - fig: Figure containing the plotted text.
        """
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, text, fontsize=fontsize, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        fig.suptitle(title, fontsize=fontsize+2)

        return self._prepare_plot(fig, None, fontsize, show_plot)

    def plot_evaluation_metrics(self, average='macro', title='Evaluation Metrics', fontsize=12, show_plot=True):
        """ 
        Plot the evaluation metrics of the model.

        Args:
        - average (str, optional): Type of averaging to use for precision, recall, and F1 score.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size for text in the plot.
        - show_plot (bool, optional): Whether to display the plot.

        Returns:
        - fig: Figure of the evaluation metrics.
        """
        
        metrics = self.calculate_evaluation_metrics(average=average)
        metrics_text = '\n'.join([f"{metric.capitalize()}: {value:.2f}" for metric, value in metrics.items()])
        return self.plot_text(metrics_text, title=title, fontsize=fontsize, show_plot=show_plot)