import tensorflow as tf
import numpy as np
import random
import cv2

from source.preprocessing.helpers.for_steps.step_base import StepBase
from source.preprocessing.helpers.for_steps.step_utils import correct_image_tensor_shape


class RandomPerspectiveTransformer(StepBase):
    """
    A preprocessing step that applies a perspective transformation to an image tensor.
    This transformation simulates a change in the viewpoint.
    """
    arguments_datatype = {'warp_scale': float, 'seed': int}
    name = 'Random Perspective Transformer'

    def __init__(self, warp_scale=0.2, seed=None):
        """
        Initializes the RandomPerspectiveTransformer object for integration into an image
        preprocessing pipeline.

        Args:
            warp_scale (float): Factor to scale the maximum warp intensity. Default is 0.2.
            seed (int): Random seed for reproducibility. Default is None.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        height, width, _ = image_nparray.shape
        warp_intensity = int(min(height, width) * self.parameters['warp_scale'])
        random.seed(self.parameters['seed'])

        src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
        dst_points = np.float32([
            [random.randint(-warp_intensity, warp_intensity),
                random.randint(-warp_intensity, warp_intensity)],
            [width - 1 - random.randint(-warp_intensity, warp_intensity),
                random.randint(-warp_intensity, warp_intensity)],
            [random.randint(-warp_intensity, warp_intensity),
                height - 1 - random.randint(-warp_intensity, warp_intensity)],
            [width - 1 - random.randint(-warp_intensity, warp_intensity),
                height - 1 - random.randint(-warp_intensity, warp_intensity)]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(image_nparray, matrix, (width, height))
        image_tensor = tf.convert_to_tensor(warped_image, dtype=self.output_datatype)
        return correct_image_tensor_shape(image_tensor)


if __name__ == '__main__':
    step = RandomPerspectiveTransformer()
    print(step.get_step_json_representation())
