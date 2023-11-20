import cv2
import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape


class AverageBlurFilter(StepBase):
    """ A preprocessing step that applies average blur filter to an image."""
    arguments_datatype = {'kernel_size': (int,int)}
    name = 'Average Blur Filter'

    def __init__(self, kernel_size=(8,8)):
        """ Initializes the `AverageBlurFilter` object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')
        
        ksize = self.params['kernel_size'] 
        cv_blurred_image = cv2.blur(cv_img, ksize)

        tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
        tf_blurred_image = correct_tf_image_shape(tf_blurred_image)

        return (tf_blurred_image, tf_target)
    

if __name__ == '__main__':
    step = AverageBlurFilter()
    print(step.get_step_json_representation())
    


