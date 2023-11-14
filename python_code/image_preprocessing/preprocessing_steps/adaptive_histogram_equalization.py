import cv2
import tensorflow as tf

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape


class AdaptiveHistogramEqualization(StepBase):

    arguments_datatype = {'clip_limit': float, 'tile_gridsize': (int, int)}
    name = 'Adaptive Histogram Equalization'

    def __init__(self, clip_limit=2.0, tile_gridsize=(8,8)):
        super().__init__(locals())

    @StepBase._py_function_decorator
    def process_step(self, tf_image, tf_target):

        cv_img = tf_image.numpy().astype('uint8')

        channels = cv2.split(cv_img)

        clahe = cv2.createCLAHE(clipLimit=self.params['clip_limit'], tileGridSize=self.params['tile_gridsize'])

        clahe_channels = [clahe.apply(ch) for ch in channels]

        cv2_clahe_image = cv2.merge(clahe_channels)

        tf_clahe_image = tf.convert_to_tensor(cv2_clahe_image, dtype=tf.uint8)

        tf_clahe_image = correct_tf_image_shape(tf_clahe_image)

        return (tf_clahe_image, tf_target)
    

if __name__ == '__main__':
    step = AdaptiveHistogramEqualization()
    print(step)
    



