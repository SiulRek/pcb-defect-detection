import cv2
import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape


class MedianBlurFilter(StepBase):
    """ A preprocessing step that applies median filter to an image."""
    arguments_datatype = {'kernel_size': int}
    name = 'Median Blur Filter'

    def __init__(self, kernel_size=5):
        """ Initializes the `MedianBlurFilter` object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')
        
        ksize = self.params['kernel_size'] 
        cv_blurred_image = cv2.medianBlur(cv_img, ksize)

        tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
        tf_blurred_image = correct_tf_image_shape(tf_blurred_image)

        return (tf_blurred_image, tf_target)
    

if __name__ == '__main__':
    step = MedianBlurFilter()
    print(step)
    


