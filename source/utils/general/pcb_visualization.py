"""
Disclaimer: The classes in this modules where used for visualization puproses
during the development of the Image Preprocessing Framework. They are not used
for plotting purposes in model development.
 """

from abc import ABC

import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf


class PCBVisualizerBase(ABC):
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

    def _get_defect_name(self, target):

        if target == 0:
            name = "no_defect"
        elif target == 1:
            name = "missing_hole"
        elif target == 2:
            name = "mouse_bite"
        elif target == 3:
            name = "open_circuit"
        elif target == 4:
            name = "short"
        elif target == 5:
            name = "spur"
        elif target == 6:
            name = "spurious_copper"
        else:
            name = "unknown"

        return name


class PCBVisualizerforTF(PCBVisualizerBase):
    """ PCB Visualizer during image processing. Pass slice from tensorflow images as
    input parameter. """

    def plot_images(self, image_tf_dataset, title="Images"):
        """
        Plots 9 images from the given TensorFlow dataset.

        Parameters:
            - image_tf_dataset: TensorFlow dataset containing the images.
            - Title: Plot title. Defaults to 'Images
        """

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        for i, image in enumerate(image_tf_dataset.take(9)):

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

    def plot_histograms(self, image_tf_dataset, title="Histograms", bins=20):
        """
        Plots histograms for 9 images from the given TensorFlow dataset.

        Parameters:
            - image_tf_dataset: TensorFlow dataset containing the images.
            - title: Plot title. Defaults to 'Histogram'.
            - bins: Number of bins in the histograms. Defaults to 20.
        """

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        for i, image in enumerate(image_tf_dataset.take(9)):

            if image.shape[-1] == 3:
                image = tf.image.rgb_to_grayscale(image)

            image_flattened = tf.reshape(image, [-1])

            axes[i].hist(image_flattened.numpy(), bins=bins, color="blue", alpha=0.7)
            axes[i].set_xlim([0, 256])
            axes[i].set_xlabel("Color Values", labelpad=10)
            axes[i].set_ylabel("Frequencies", labelpad=10)

        self._generate_plot(fig, title, y_title=1.03, wspace=0.4, hspace=0.4)

    def plot_frequency_spectrum_3D(
        self, image_tf_dataset, title="Frequency Spectrum of Images", cmap="viridis"
    ):
        """
        Plots the 3D frequency spectrum of up to 9 images from a TensorFlow
        dataset.

        Parameters:
            - image_tf_dataset: TensorFlow dataset containing images.
            - title: Title for the plot (default is "Frequency Spectrum of
                Images").
            - cmap: Colormap used for 3D surface (default is 'viridis').
        """

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        for i, image in enumerate(image_tf_dataset.take(9)):
            ax = fig.add_subplot(3, 3, i + 1, projection="3d")

            magnitude_spectrum = self._compute_image_fft(image)

            # Create a meshgrid for 3D plotting
            x = np.linspace(
                0, magnitude_spectrum.shape[0] - 1, magnitude_spectrum.shape[0]
            )
            y = np.linspace(
                0, magnitude_spectrum.shape[1] - 1, magnitude_spectrum.shape[1]
            )
            X, Y = np.meshgrid(y, x)

            ax.plot_surface(X, Y, np.log(1 + magnitude_spectrum), cmap=cmap)

            # Adjust axis labels for better visibility
            ax.set_xlabel("Frequency X", labelpad=10)
            ax.set_ylabel("Frequency Y", labelpad=10)
            ax.set_zlabel("Magnitude", labelpad=5)

            # Increase tick label size
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.tick_params(axis="both", which="major", labelsize=8)

        fig.subplots_adjust(
            left=0.10, right=0.90, bottom=0.10, top=0.90, wspace=0.70, hspace=0.70
        )
        plt.tight_layout()
        self.last_fig = fig
        if self.show_plot:
            plt.show()

    def _compute_image_fft(self, image):

        # image_conv = tf.cast(image, dtype=tf.int16)
        image_np = tf.image.rgb_to_grayscale(image).numpy().squeeze()

        f_transform = np.fft.fft2(image_np)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        return magnitude_spectrum


class PCBVisualizerforCV2(PCBVisualizerBase):
    """ PCB Visualizer during image processing. Pass list of canvas images as input
    parameter. """

    def plot_images(self, image_list, title="Images"):
        """
        Plots up to 9 images from the given list of images.

        Parameters:
            - image_list: List containing the canvas images.
            - title: Plot title. Defaults to 'Images'.
        """

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        num_images = min(9, len(image_list))

        for i in range(num_images):

            # Convert the BGR image from cv2 to RGB for displaying using matplotlib
            image = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)

            if image.ndim == 3 and image.shape[2] == 1:
                axes[i].imshow(np.squeeze(image), cmap="gray")  # use the gray colormap
            else:
                axes[i].imshow(image)

            axes[i].axis("off")

        self._generate_plot(fig, title, y_title=0.95)

    def plot_image_comparison(
        self, original_img_list, processed_img_list, index, title="Image Comparison"
    ):
        """
        Plots a comparison of original and processed image.

        Parameters:
            - original_img_list: List containing the original canvas image.
            - processed_img_list: List containing the processed canvas
                image.
            - index: The index number of the images in the corresponding
                list.
            - Title: Plot title. Defaults to 'Image Comparison'.
        """

        original_image = cv2.cvtColor(original_img_list[index], cv2.COLOR_BGR2RGB)
        processed_image = cv2.cvtColor(processed_img_list[index], cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.ravel()

        images = [original_image, processed_image]

        for i, image in enumerate(images):

            if len(image.shape) == 3 and image.shape[2] == 1:
                axes[i].imshow(np.squeeze(image), cmap="gray")  # use the gray colormap
            else:
                axes[i].imshow(image)

            axes[i].axis("off")

        self._generate_plot(fig, title)

    def plot_detected_corners(self, image_list, corners_list, title="Detected Corners"):
        """
        Plots up to 9 images with their detected corners from the given list of
        images.

        Parameters:
            - images: List containing the canvas images.
            - corners_list: List of lists containing the detected corners
                for each image.
            - title: Plot title. Defaults to 'Detected Corners'.
        """

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(9):

            image = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
            axes[i].imshow(image)

            # Plot the detected corners on the image

            for j, corner in enumerate(corners_list[i]):
                axes[i].scatter(corner[1], corner[0], s=100, c="red", marker="o")
                axes[i].text(corner[1], corner[0], str(j), color="orange")

            axes[i].set_title(f"Detected Corners: {len(corners_list[i])}", color="red")

            axes[i].axis("off")

        self._generate_plot(fig, title, y_title=0.95)
