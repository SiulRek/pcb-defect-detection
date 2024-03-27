import cv2
import numpy as np
from source.image_preprocessing.preprocessing_steps.step_base import StepBase
import random


class RandomSharpening(StepBase):
    """
    A data augmentation step that applies random sharpening to an image.
    """

    arguments_datatype = {'min_intensity': float, 'max_intensity': float}
    name = 'Random Sharpening'

    def __init__(self, min_intensity=0.5, max_intensity=2.0):
        """
        Initializes the RandomSharpening object for integration in an image preprocessing pipeline.
        
        Args:
            min_intensity (float): Minimum intensity of sharpening.
            max_intensity (float): Maximum intensity of sharpening.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        intensity = random.uniform(self.parameters['min_intensity'], self.parameters['max_intensity'])

        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]]) * intensity
        kernel[1, 1] += 1

        sharpened_image = cv2.filter2D(image_nparray, -1, kernel)
        sharpened_image = np.clip(sharpened_image, 0, 255)

        return sharpened_image.astype(np.uint8)

if __name__ == '__main__':
    step = RandomSharpening()
    print(step.get_step_json_representation())
