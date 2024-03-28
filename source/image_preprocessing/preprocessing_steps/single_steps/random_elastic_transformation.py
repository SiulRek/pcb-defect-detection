import cv2
import numpy as np

from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class RandomElasticTransformer(StepBase):
    """
    A data augmentation step that applies a Random Elastic Transformer to an image.
    """

    arguments_datatype = {'alpha': float, 'sigma': float, 'seed': int}
    name = 'Random Elastic Transformer'

    def __init__(self, alpha=34, sigma=4, seed=42):
        """
        Initializes the RandomElasticTransformer object for integration into an image preprocessing pipeline.
        
        Args:
            alpha (float): Intensity of the transformation. Default is 34.
            sigma (float): Standard deviation of the Gaussian filter. Default is 4.
            seed (int): Random seed for reproducibility. Default is 42.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        row, col, _ = image_nparray.shape

        np.random.seed(self.parameters['seed'])  # Set the random seed

        dx = np.random.uniform(-1, 1, size=(row, col)) * self.parameters['alpha']
        dy = np.random.uniform(-1, 1, size=(row, col)) * self.parameters['alpha']

        kernel_size = int(6 * self.parameters['sigma']) + 1
        kernel_size = (kernel_size, kernel_size)  # Making it a tuple

        sdx = cv2.GaussianBlur(dx, kernel_size, 0)
        sdy = cv2.GaussianBlur(dy, kernel_size, 0)
        
        x, y = np.meshgrid(np.arange(col), np.arange(row))
        map_x, map_y = np.float32(x + sdx), np.float32(y + sdy)

        elastic_transformed_image = cv2.remap(image_nparray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return elastic_transformed_image

if __name__ == '__main__':
    step = RandomElasticTransformer()
    print(step.get_step_json_representation())
