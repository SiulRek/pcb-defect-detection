import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import seaborn as sns
from sklearn.metrics import confusion_matrix


class ImageClassifierVisualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_predictions_prepared = False

    def _unbatch_dataset(self, dataset):
        try:
            return dataset.unbatch()
        except ValueError:
            return dataset

    def _prepare_plot(self, title=None, fontsize=12, show_plot=True):  
        plt.tight_layout()  
        if title:
            plt.suptitle(title, fontsize=fontsize)
        plt.tight_layout()
        if show_plot:
            plt.show()
    
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
        boolean_mask = np.array([np.argmax(label) in class_indices for _, label in np_dataset])
        filtered_data = np_dataset[boolean_mask]
        return filtered_data, boolean_mask
    
    def _shuffle_in_unison(self, np_dataset1, np_dataset2):
        assert len(np_dataset1) == len(np_dataset2), "Datasets must be the same length"
        
        p = np.random.permutation(len(np_dataset1))
        return np_dataset1[p], np_dataset2[p]

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
        - plt: Plot of the images.
        """
        np_dataset = self._prepare_dataset(combined_dataset)

        _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes = axes.flatten()
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
                label_name = self.class_names[np.argmax(label)]
                ax.set_title(label_name, fontsize=fontsize)

        self._prepare_plot(title, fontsize, show_plot)
    
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
        - plt: Plot of the images.
        """
        np_dataset = self._prepare_dataset(combined_dataset)
        filtered_dataset, _ = self._filter_np_dataset(np_dataset, classes)
        plot = self.plot_images(filtered_dataset, n_rows=n_rows, n_cols=n_cols, title=title, fontsize=fontsize, show_plot=show_plot)
        return plot
    
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
        - plt: Plot of the images.
        """
        np_dataset_1 = self._prepare_dataset(combined_dataset_1, shuffle=False)
        np_dataset_2 = self._prepare_dataset(combined_dataset_2, shuffle=False)
        np_dataset_1, np_dataset_2 = self._shuffle_in_unison(np_dataset_1, np_dataset_2)

        _, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3))
        axes = axes.flatten()
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
                subtitle += self.class_names[np.argmax(label)]
            ax.set_title(subtitle, fontsize=fontsize)

        self._prepare_plot(title, fontsize, show_plot)

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
        - plt: Plot of the images.
        """
        np_dataset_1 = self._prepare_dataset(combined_dataset_1, shuffle=False)
        np_dataset_2 = self._prepare_dataset(combined_dataset_2, shuffle=False)
        filtered_dataset_1, _ = self._filter_np_dataset(np_dataset_1, classes)
        filtered_dataset_2, _ = self._filter_np_dataset(np_dataset_2, classes)
        filtered_dataset_1, filtered_dataset_2 = self._shuffle_in_unison(filtered_dataset_1, filtered_dataset_2)
        plot = self.plot_image_comparisons(filtered_dataset_1, filtered_dataset_2, n_rows=n_rows, n_cols=n_cols, title=title, fontsize=fontsize, show_plot=show_plot)
        return plot

    def prepare_model_predictions(self, model, dataset):
        """ Prepares model predictions and dataset for visualization.

        Args:
        - model (tf.keras.Model): Trained TensorFlow model.
        - dataset (tf.data.Dataset): TensorFlow dataset consisting of tuples (image, label).

        Returns:
        - None
        """
        self.model = model
        self.predictions = model.predict(dataset)
        self.np_dataset = self._prepare_dataset(dataset, shuffle=True)
        self.model_predictions_prepared = True

    def _plot_results(self, np_dataset, predictions, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        _, axes = plt.subplots(n_rows, n_cols * (2 if prediction_bar else 1), figsize=(n_cols * (6 if prediction_bar else 3), n_rows * 3))
        axes = axes.flatten()

        for i in range(n_rows * n_cols):
            img_ax = axes[i * (2 if prediction_bar else 1)]
            image, true_label = np_dataset[i]
            predicted_probs = predictions[i]
            predicted_label_index = np.argmax(predicted_probs)
            true_label_index = np.argmax(true_label)

            img_ax.imshow(image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
            img_ax.axis('off')

            title_color = 'green' if predicted_label_index == true_label_index else 'red'
            bar_colors = ['green' if j == true_label_index else 'blue' for j in range(len(predicted_probs))]
            if predicted_label_index != true_label_index:
                bar_colors[predicted_label_index] = 'red'

            true_label_name = self.class_names[true_label_index]
            predicted_label_name = self.class_names[predicted_label_index]
            img_ax.set_title(f'Predicted: {predicted_label_name}\nTrue: {true_label_name}', fontsize=fontsize, color=title_color)

            if prediction_bar:
                bar_ax = axes[i * 2 + 1]
                bar_ax.bar(range(len(predicted_probs)), predicted_probs, color=bar_colors)
                bar_ax.set_xticks(range(len(self.class_names)))
                bar_ax.set_xticklabels(self.class_names, rotation=90)
                bar_ax.set_ylim(0, 1)

        self._prepare_plot(title, fontsize, show_plot)

    def plot_results(self, n_rows=3, n_cols=3, title=None, fontsize=12, prediction_bar=True, show_plot=True):
        """ Plots a set of images along with their predicted and true labels, with an optional prediction bar.

        Args:
        - n_rows (int): Number of rows in the grid.
        - n_cols (int): Number of columns in the grid.
        - title (str, optional): Title of the plot.
        - fontsize (int, optional): Font size of the labels.
        - prediction_bar (bool): Whether to add a prediction bar.
        - show_plot (bool): Whether to display the plot.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")
        
        self._plot_results(self.np_dataset, self.predictions, n_rows=n_rows, n_cols=n_cols, title=title, 
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
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")
        
        filtered_dataset, boolean_mask = self._filter_np_dataset(self.np_dataset, classes)
        filtered_predictions = self.predictions[boolean_mask]
        self._plot_results(filtered_dataset, filtered_predictions, n_rows=n_rows, n_cols=n_cols, title=title, 
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
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")
        
        boolean_mask = np.argmax(self.predictions, axis=1) != np.argmax([label for _, label in self.np_dataset], axis=1)
        filtered_dataset = self.np_dataset[boolean_mask]
        filtered_predictions = self.predictions[boolean_mask]
        self._plot_results(filtered_dataset, filtered_predictions, n_rows=n_rows, n_cols=n_cols, title=title, 
                           fontsize=fontsize, prediction_bar=prediction_bar, show_plot=show_plot)
        
    def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, fontsize=12, show_plot=True):
        """ Plots the confusion matrix for the model's predictions.

        Args:
        - normalize (bool, optional): Whether to normalize the confusion matrix.
        - title (str, optional): Title of the plot.
        - cmap (matplotlib.colors.Colormap, optional): Colormap of the plot.
        - fontsize (int, optional): Font size of the labels.
        - show_plot (bool): Whether to display the plot.
        """
        if not self.model_predictions_prepared:
            raise ValueError("Model predictions have not been prepared. Please call prepare_model_predictions first.")

        true_labels = np.argmax([label for _, label in self.np_dataset], axis=1)
        predicted_labels = np.argmax(self.predictions, axis=1)

        cm = confusion_matrix(true_labels, predicted_labels)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('True label', fontsize=fontsize)
        plt.xlabel('Predicted label', fontsize=fontsize)

        self._prepare_plot(title=title, fontsize=fontsize, show_plot=show_plot)