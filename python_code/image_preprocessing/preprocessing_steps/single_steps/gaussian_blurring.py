import cv2
import tensorflow as tf

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape


class GaussianBlurring(StepBase):
    """  #TODO A preprocessing step that applies Contrast Limited Gaussian Blurring to an image."""
    arguments_datatype = {'kernel_size': (int,int), 'sigma': float}
    name = 'Gaussian Blurring'

    def __init__(self, kernel_size=(5,5), sigma=0.3):
        """ Initializes the GaussianBlurring object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')

        k = self.params['kernel_size']
        sigma = self.params['sigma']
        cv_blurred_image = cv2.GaussianBlur(cv_img, k, sigma)

        tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
        tf_blurred_image = correct_tf_image_shape(tf_blurred_image)

        return (tf_blurred_image, tf_target)
    

if __name__ == '__main__':
    step = GaussianBlurring()
    print(step)
    


