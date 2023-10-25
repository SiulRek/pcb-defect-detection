import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf

class PCBVisualizer:
    """ PCB Visualizer during image processing. Pass slice from tensorflow images as input parameter. """

    def __init__(self):
        self.last_plot = None  

    def plot_images(self, image_tf_dataset, title="Images"):
        """
        Plots 9 images from the given TensorFlow dataset.
        
        Parameters:
        - image_tf_dataset: TensorFlow dataset containing the images.
        - Title: Plot title. Defaults to 'Images
        """
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()
        
        for i, image_data in enumerate(image_tf_dataset.take(9)):
            
            if len(image_data[0].shape) == 3 and image_data[0].shape[2] == 1:
                axes[i].imshow(tf.squeeze(image_data[0]).numpy(), cmap='gray')  # use the gray colormap       
            else:
                axes[i].imshow(image_data[0].numpy())
            
            axes[i].axis('off')
            
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.show()

        self.last_plot = plt.gcf()  

    def plot_histograms(self, image_tf_dataset, title='Histograms', bins=20):
        """
        Plots histograms for 9 images from the given TensorFlow dataset.
        
        Parameters:
        - image_tf_dataset: TensorFlow dataset containing the images.
        - title:    Plot title. Defaults to 'Histogram'.
        - bins: Number of bins in the histograms. Defaults to 20.
        """
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()
        
        for i, image_data in enumerate(image_tf_dataset.take(9)):

            if image_data[0].shape[-1] == 3:
                image = tf.image.rgb_to_grayscale(image_data[0])
            else: 
                image = image_data[0]
            
            image_flattened = tf.reshape(image, [-1])
            
            axes[i].hist(image_flattened.numpy(), bins=bins, color='blue', alpha=0.7)
            axes[i].set_xlim([0, 256])
            axes[i].set_xlabel("Color Values", labelpad=10)
            axes[i].set_ylabel("Frequencies", labelpad=10)

        fig.suptitle(title, fontsize=20, fontweight='bold', y=1.03)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()

        self.last_plot = plt.gcf()  

    def plot_frequency_spectrum_3D(self, image_tf_dataset, title="Frequency Spectrum of Images", cmap='viridis'):

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        for i, image_data in enumerate(image_tf_dataset.take(9)):
            ax = fig.add_subplot(3, 3, i+1, projection='3d')
            
            magnitude_spectrum = self._compute_image_fft(image_data[0])

            # Create a meshgrid for 3D plotting
            x = np.linspace(0, magnitude_spectrum.shape[0]-1, magnitude_spectrum.shape[0])
            y = np.linspace(0, magnitude_spectrum.shape[1]-1, magnitude_spectrum.shape[1])
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
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90, wspace=0.70, hspace=0.70)
        plt.tight_layout()
        plt.show()

        self.last_plot = plt.gcf()  

    def _compute_image_fft(self, image): 
        
        #image_conv = tf.cast(image, dtype=tf.int16)
        image_np = tf.image.rgb_to_grayscale(image).numpy().squeeze()

        f_transform = np.fft.fft2(image_np)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        return magnitude_spectrum
        
    def save_plot_to_file(self, filename):
        if self.last_plot:
            self.last_plot.savefig(filename)
        else:
            print("No plot to save!")
