import cv2
import numpy as np
import random

from source.preprocessing.helpers.step_base import StepBase


class RandomRotator(StepBase):
    """
    A preprocessing step that applies Random Rotator to an image within a specified angle range.
    """
    arguments_datatype = {'angle_range': (int, int), 'seed': int}
    name = 'Random Rotator'

    def __init__(self, angle_range=(-90, 90), seed=42):
        """
        Initializes the RandomRotator object that can be integrated in an image preprocessing
            pipeline.

        Args:
            angle_range (tuple): Tuple of two integers specifying the range of angles for rotation.
                                 For example, (-90, 90) allows rotations between -90 and 90 degrees.
            seed (int): Random seed for reproducible rotations. Default is 42.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        random.seed(self.parameters['seed'])

        angle = random.randint(*self.parameters['angle_range'])
        height, width = image_nparray.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_image = cv2.warpAffine(image_nparray, rotation_matrix, (width, height))
        return rotated_image


if __name__ == '__main__':
    step = RandomRotator()
    print(step.get_step_json_representation())
