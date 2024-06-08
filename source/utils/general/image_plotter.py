"""
Disclaimer: The classes in this modules where used for visualization puproses
during the development of the Image Preprocessing Framework. They are not used
for plotting purposes in model development.
 """

from abc import ABC

import matplotlib.pyplot as plt
import tensorflow as tf


class ImagePlotterBase(ABC):
    """ This class represents the base for the PCBVisualizer child classes. """

    def __init__(self, show_plot=True):
        self.last_fig = None
        self.show_plot = show_plot

    def save_plot_to_file(self, filename):
        if self.last_fig:
            self.last_fig.savefig(filename)
        else:
            print("No plot to save!")
        plt.close()

    def _generate_plot(self, fig, title, y_title=0.95, wspace=0.01, hspace=0.01):
        fig.suptitle(title, fontsize=20, fontweight="bold", y=y_title)
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        self.last_fig = fig
        if self.show_plot:
            plt.show()


class ImagePlotter(ImagePlotterBase):
    """ ImagePlotter during image processing. Pass slice from tensorflow images as
    input parameter. """

    def plot_images(self, image_tf_dataset, title="Images"):
        """
        Plots 9 images from the given TensorFlow dataset.

        Parameters:
            - image_tf_dataset: TensorFlow dataset containing the images.
            - Title: Plot title. Defaults to 'Images
        """

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.ravel()

        for i, image in enumerate(image_tf_dataset.take(4)):

            if len(image.shape) == 3 and image.shape[2] == 1:
                axes[i].imshow(
                    tf.squeeze(image).numpy(), cmap="gray"
                )  # use the gray colormap
            else:
                axes[i].imshow(image.numpy())

            axes[i].axis("off")

        self._generate_plot(fig, title)

    def plot_image_comparison(
        self, original_tf_dataset, processed_tf_dataset, index, title=""
    ):
        """
        Plots a comparison of original and processed image.

        Parameters:
            - original_tf_dataset: TensorFlow dataset containing the origina
                image.
            - compare_tf_dataset: TensorFlow dataset containing the compare
                image.
            - index: The index number of the images in the corresponding
                dataset.
            - Title: Plot title. Defaults to 'Image Comparison'.
        """

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.ravel()

        image_data_org = original_tf_dataset.skip(index).take(1)
        image_data_prc = processed_tf_dataset.skip(index).take(1)

        for i, take_object in enumerate([image_data_org, image_data_prc]):
            for image in take_object:
                if len(image.shape) == 3 and image.shape[2] == 1:
                    axes[i].imshow(
                        tf.squeeze(image).numpy(), cmap="gray"
                    )  # use the gray colormap
                else:
                    axes[i].imshow(image.numpy())

                axes[i].axis("off")

        if title == "":
            title = "Compare Images"

        self._generate_plot(fig, title)
