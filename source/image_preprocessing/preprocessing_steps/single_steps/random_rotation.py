import cv2
import numpy as np
from source.image_preprocessing.preprocessing_steps.step_base import StepBase


class RandomRotation(StepBase):
    """
    A preprocessing step that applies random rotation to an image within a specified angle range.
    """

    arguments_datatype = {'angle_range': (int, int)}
    name = 'Random Rotation'

    def __init__(self, angle_range=(-90, 90)):
        """
        Initializes the RandomRotation object that can be integrated in an image preprocessing pipeline.
        
        Args:
            angle_range (tuple): A tuple of two integers specifying the range of angles for rotation. 
                                 For example, (-90, 90) allows rotations between -90 and 90 degrees.
        """
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        angle = np.random.randint(*self.parameters['angle_range'])
        height, width = image_nparray.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_image = cv2.warpAffine(image_nparray, rotation_matrix, (width, height))
        return rotated_image

if __name__ == '__main__':
    step = RandomRotation()
    print(step.get_step_json_representation())
