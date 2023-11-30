#TODO update according to changes in decorator.
"""
This module contains a thorough test suite for validating the StepBase class in the image preprocessing module. It includes tests for both TensorFlow (TfTestStep) and Python (PyTestStep) image preprocessing steps to ensure proper initialization, functionality, and pipeline compatibility. The tests cover a wide range of topics, including image shape transformation, custom object equality logic, and proper JSON representation. The module also tests the efficacy of decorators (_tensor_pyfunc_wrapper and _nparray_pyfunc_wrapper) in processing image datasets. These tests are critical for ensuring the integrity and dependability of the pipeline's image preprocessing steps.
"""

import os
import unittest

import tensorflow as tf
import cv2

from source.load_raw_data.kaggle_dataset import load_tf_record
from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_image_tensor_shape
from source.utils import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/image_preprocessing/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class TfTestStep(StepBase):

    arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
    name = 'Test_Step'
    
    def __init__(self, param1=10 , param2=(10,10), param3=True):
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
        image_grayscale_tensor = correct_image_tensor_shape(image_grayscale_tensor)
        return image_grayscale_tensor

class PyTestStep(StepBase):
    
    arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
    name = 'Test_Step'
    
    def __init__(self, param1=10 , param2=(10,10), param3=True):
        super().__init__(locals())

    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        blurred_image = cv2.GaussianBlur(image_nparray, ksize=(5,5), sigmaX=2)  # Randomly choosen action.
        blurred_image_tensor = tf.convert_to_tensor(blurred_image, dtype=tf.uint8)
        image_grayscale_tensor = tf.image.rgb_to_grayscale(blurred_image_tensor)
        processed_img = (image_grayscale_tensor.numpy()).astype('uint8')
        return (processed_img)
     # Note in real usage conversion of np.array to tensor and viceversa in one process_step is not recommended.
    

class TestStepBase(unittest.TestCase):
    """    Test suite for validating the functionality of the preprocessing steps parent class 'StepBase' in the image preprocessing module.
    
    This suite includes tests to verify the correct initialization of preprocessing steps,  the proper function
    of both TensorFlow and Python-based image preprocessing steps. Additionally, it ensures that images
    maintain correct shape transformations throughout the preprocessing pipeline and that the custom object
    equality logic behaves as expected. The example test steps `TfTestStep` and `PyTestStep` make a simple conversion of the images from RGB to Grayscale, that can be easily verified.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_dataset = load_tf_record().take(9)
        cls.logger = TestResultLogger(LOG_FILE, 'Step Base Test')
    
    def setUp(self):
        self.local_vars = {'param1': 10, 'param2': (10,10), 'param3': True}
        self.tf_preprocessing_step = TfTestStep(**self.local_vars)
        self.py_preprocessing_step = PyTestStep(**self.local_vars)

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)
        
    def test_initialization(self):
        self.assertEqual(self.tf_preprocessing_step.name, "Test_Step")
        self.assertEqual(self.tf_preprocessing_step.params, {'param1': 10, 'param2': (10,10), 'param3': True})

    def test_correct_shape_gray(self):
        image_tensor = list(TestStepBase.image_dataset.take(1))[0][0]
        image_grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
        reshaped_image = correct_image_tensor_shape(image_grayscale_tensor)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 1])

    def test_correct_shape_rgb(self):
        image_tensor = list(TestStepBase.image_dataset.take(1))[0][0]
        reshaped_image = correct_image_tensor_shape(image_tensor)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 3])
    
    def _remove_new_lines_and_spaces(self, string):
        string = string.replace('\n','')
        string = string.replace(' ','')
        return string
    
    def test_get_step_json_representation(self):
        json_repr_output = self.tf_preprocessing_step.get_step_json_representation()
        json_repr_expected = '"Test_Step": {"param1": [10], "param2": [[10,10]], "param3": [true]}'
        json_repr_output = self._remove_new_lines_and_spaces(json_repr_output)
        json_repr_expected = self._remove_new_lines_and_spaces(json_repr_expected)
        self.assertEqual(json_repr_output,json_repr_expected)

    def test_tensor_pyfunc_wrapper(self):
        tf_dataset = self.tf_preprocessing_step.process_step(self.image_dataset)
        image_tensor = list(tf_dataset.take(1))[0][0]
        self.assertEqual(image_tensor.shape, [2464, 3056, 1])

    def test_nparray_pyfunc_wrapper(self):
        tf_dataset = self.py_preprocessing_step.process_step(self.image_dataset)
        image_tensor = list(tf_dataset.take(1))[0][0]
        self.assertEqual(image_tensor.shape, [2464, 3056, 1])
    
    def test_equal_objects(self):
        self.assertEqual(self.py_preprocessing_step, self.tf_preprocessing_step)

    def test_not_equal_objects(self):
        local_vars = { 'param1': 20, 'param2': (20,20), 'param3': False}
        tf_preprocessing_step = TfTestStep(**local_vars)
        self.assertNotEqual(self.py_preprocessing_step, tf_preprocessing_step)


if __name__ == '__main__':
    unittest.main()
