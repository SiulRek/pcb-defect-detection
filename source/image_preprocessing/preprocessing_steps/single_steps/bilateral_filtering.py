import cv2
import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape


class BilateralFilter(StepBase):
    """ A preprocessing step that applies bilateral filter to an image."""
    arguments_datatype = {
        'diameter': int,
        'sigma_color':float, 
        'sigma_space':float
        }
    name = 'Bilateral Filter'

    def __init__(self, diameter=9, sigma_color=75, sigma_space=75):
        """ Initializes the `BilateralFilter` object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')
        
        cv_blurred_image = cv2.bilateralFilter(
            src=cv_img,
            d=self.params['diameter'], 
            sigmaColor=self.params['sigma_color'], 
            sigmaSpace=self.params['sigma_space'])

        tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
        tf_blurred_image = correct_tf_image_shape(tf_blurred_image)

        return (tf_blurred_image, tf_target)
    

if __name__ == '__main__':
    step = BilateralFilter()
    print(step.get_step_json_representation())
    


