import cv2
import tensorflow as tf

from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape


class GlobalHistogramEqualizer(StepBase):
    """  A preprocessing step that applies Contrast Limited Global Histogram Equalization to an image."""
    arguments_datatype = {}
    name = 'Global Histogram Equalization'

    def __init__(self):
        """ Initializes the GlobalHistogramEqualizer object that can be integrated in an image preprocessing pipeline."""
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')

        channels = cv2.split(cv_img)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]  
        cv2_eq_image = cv2.merge(eq_channels)

        tf_eq_image = tf.convert_to_tensor(cv2_eq_image, dtype=tf.uint8)
        tf_eq_image = correct_tf_image_shape(tf_eq_image)

        return (tf_eq_image, tf_target)
    

if __name__ == '__main__':
    step = GlobalHistogramEqualizer()
    print(step)
    


