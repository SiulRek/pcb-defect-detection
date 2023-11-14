import unittest

import tensorflow as tf
import cv2

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape
from python_code.load_raw_data.kaggle_dataset import load_tf_record


class TestStepBase(unittest.TestCase):
    """    Test suite for validating the functionality of the preprocessing steps parent class 'StepBase' in the image preprocessing module.
    
    This suite includes tests to verify the correct initialization of preprocessing steps,  the proper function
    of both TensorFlow and Python-based image preprocessing steps. Additionally, it ensures that images
    maintain correct shape transformations throughout the preprocessing pipeline and that the custom object
    equality logic behaves as expected. The example test steps `TfTestStep` and `PyTestStep` make a simple conversion of the images from RGB to Grayscale, that can be easily verified.
    """
    class TfTestStep(StepBase):

        arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
        name = 'Test_Step'
        
        def __init__(self, param1=10 , param2=(10,10), param3=True):
            super().__init__(locals())

        @StepBase._tf_function_decorator
        def process_step(self, tf_image, tf_target):
            tf_image_grayscale = tf.image.rgb_to_grayscale(tf_image)
            correct_tf_image_shape(tf_image_grayscale)
            return tf_image_grayscale, tf_target

    class PyTestStep(StepBase):
        
        arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
        name = 'Test_Step'
        
        def __init__(self, param1=10 , param2=(10,10), param3=True):
            super().__init__(locals())

        @StepBase._py_function_decorator
        def process_step(self, tf_image, tf_target):
            cv_img = (tf_image.numpy()).astype('uint8')
            cv_blurred_image = cv2.GaussianBlur(cv_img, ksize=(5,5), sigmaX=2)  # Randomly choosen action.
            tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
            tf_image_grayscale = tf.image.rgb_to_grayscale(tf_blurred_image)
            tf_blurred_image = correct_tf_image_shape(tf_blurred_image)
            return (tf_image_grayscale, tf_target)

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_dataset = load_tf_record().take(9)
    
    def setUp(self):
        self.local_vars = {'param1': 10, 'param2': (10,10), 'param3': True}
        self.tf_preprocessing_step = self.TfTestStep(**self.local_vars)
        self.py_preprocessing_step = self.PyTestStep(**self.local_vars)
    
    def test_initialization(self):
        self.assertEqual(self.tf_preprocessing_step.name, "Test_Step")
        self.assertEqual(self.tf_preprocessing_step.params, {'param1': 10, 'param2': (10,10), 'param3': True})

    def test_correct_shape_gray(self):
        tf_image = list(TestStepBase.image_dataset.take(1))[0][0]
        tf_image_grayscale = tf.image.rgb_to_grayscale(tf_image)
        reshaped_image = correct_tf_image_shape(tf_image_grayscale)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 1])

    def test_correct_shape_rgb(self):
        tf_image = list(TestStepBase.image_dataset.take(1))[0][0]
        reshaped_image = correct_tf_image_shape(tf_image)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 3])

    def test_tf_function_decorator(self):
        tf_dataset = self.tf_preprocessing_step.process_step(self.image_dataset)
        tf_image = list(tf_dataset.take(1))[0][0]
        self.assertEqual(tf_image.shape, [2464, 3056, 1])

    def test_py_function_decorator(self):
        tf_dataset = self.py_preprocessing_step.process_step(self.image_dataset)
        tf_image = list(tf_dataset.take(1))[0][0]
        self.assertEqual(tf_image.shape, [2464, 3056, 1])
    
    def test_equal_objects(self):
        self.assertEqual(self.py_preprocessing_step, self.tf_preprocessing_step)

    def test_not_equal_objects(self):
        local_vars = { 'param1': 20, 'param2': (20,20), 'param3': False}
        tf_preprocessing_step = self.TfTestStep(**local_vars)
        self.assertNotEqual(self.py_preprocessing_step, tf_preprocessing_step)


if __name__ == '__main__':
    unittest.main()
