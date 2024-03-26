import cv2
import numpy as np
from source.image_preprocessing.preprocessing_steps.step_base import StepBase
import random


class RandomFlipper(StepBase):
    """
    A data augmentation step that flips the image randomly in a specified direction.
    """

    arguments_datatype = {'flip_direction': str}
    name = 'Random Flipper'

    def __init__(self, flip_direction='horizontal'):
        """
        Initializes the RandomFlipper object for integration in an image preprocessing pipeline.
        
        Args:
            flip_direction (str): Direction of the potential flip. Can be 'horizontal', 'vertical', or 'both'.
        """
        super().__init__(locals())
        if flip_direction not in ['horizontal', 'vertical', 'both']:
            raise ValueError("flip_direction must be 'horizontal', 'vertical', or 'both'.")

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        flip_direction = self.parameters['flip_direction']
        do_flip = random.choice([True, False])

        if do_flip:
            if flip_direction == 'horizontal':
                flipped_image = cv2.flip(image_nparray, 1)  # Flip horizontally
            elif flip_direction == 'vertical':
                flipped_image = cv2.flip(image_nparray, 0)  # Flip vertically
            elif flip_direction == 'both':
                flipped_image = cv2.flip(image_nparray, -1) # Flip both ways
        else:
            flipped_image = image_nparray  # No flip applied

        return flipped_image

if __name__ == '__main__':
    step = RandomFlipper()
    print(step.get_step_json_representation())
