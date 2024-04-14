import os
import unittest

import tensorflow as tf
import cv2

from source.load_raw_data.kaggle_dataset import load_tf_record
from source.load_raw_data.unpack_tf_dataset import unpack_tf_dataset
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
    """    Test suite for validating the functionality of the preprocessing steps parent class `StepBase` in the image preprocessing module.
    
    This test suite is designed to validate the `StepBase` class, focusing on the correct initialization and functionality of both TensorFlow and Python-based preprocessing steps. It incorporates tests for image shape transformations, object equality logic, JSON representation of steps, wrapper functions for processing image data and output datatype handling. The suite employs TfTestStep and PyTestStep for transforming images from RGB to grayscale, a process chosen for its straightforward verification of step effectiveness. This transformation serves as a reliable indicator; if these steps work correctly, it's likely that other steps with a similar structure will function effectively as well.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        kaggle_dataset = load_tf_record().take(9)
        cls.image_dataset = unpack_tf_dataset(kaggle_dataset)[0]
        cls.logger = TestResultLogger(LOG_FILE, 'Step Base Test')
    
    def setUp(self):
        self.local_vars = {'param1': 10, 'param2': (10,10), 'param3': True}
        self.tf_preprocessing_step = TfTestStep(**self.local_vars)
        self.py_preprocessing_step = PyTestStep(**self.local_vars)

    def tearDown(self):
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def _verify_image_shapes(self, processed_images, original_images, color_channel_expected):
        for original_image, processed_image in zip(original_images, processed_images):
            self.assertEqual(processed_image.shape[:1], original_image.shape[:1]) # Check if height and width are equal.
            self.assertEqual(color_channel_expected, processed_image.shape[2])   
        
    def test_initialization(self):
        self.assertEqual(self.tf_preprocessing_step.name, "Test_Step")
        self.assertEqual(self.tf_preprocessing_step.parameters, {'param1': 10, 'param2': (10,10), 'param3': True})

    def test_correct_shape_gray(self):
        image_tensor = list(TestStepBase.image_dataset.take(1))[0]
        image_grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
        reshaped_image = correct_image_tensor_shape(image_grayscale_tensor)
        self.assertEqual(reshaped_image.shape, image_tensor.shape[:2] + [1])

    def test_correct_shape_rgb(self):
        image_tensor = list(TestStepBase.image_dataset.take(1))[0]
        image_grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
        image_rgb_tensor = tf.image.grayscale_to_rgb(image_grayscale_tensor)
        reshaped_image = correct_image_tensor_shape(image_rgb_tensor)
        self.assertEqual(reshaped_image.shape, image_tensor.shape)
    
    def _remove_new_lines_and_spaces(self, string):
        string = string.replace('\n','')
        string = string.replace(' ','')
        return string
    
    def test_get_step_json_representation(self):
        json_repr_output = self.tf_preprocessing_step.get_step_json_representation()
        json_repr_expected = '"Test_Step": {"param1": 10, "param2": [10,10], "param3": true}'
        json_repr_output = self._remove_new_lines_and_spaces(json_repr_output)
        json_repr_expected = self._remove_new_lines_and_spaces(json_repr_expected)
        self.assertEqual(json_repr_output,json_repr_expected)

    def test_tensor_pyfunc_wrapper(self):
        processed_dataset = self.tf_preprocessing_step.process_step(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, 1)

    def test_nparray_pyfunc_wrapper(self):
        processed_dataset = self.py_preprocessing_step.process_step(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, 1)

    def test_output_datatype_conversion(self):
        self.py_preprocessing_step.output_datatype = tf.uint8
        processed_dataset = self.py_preprocessing_step.process_step(self.image_dataset)
        for img in processed_dataset.take(1):
            self.assertEqual(img.dtype, tf.uint8)
        self.py_preprocessing_step.output_datatype = tf.float16
        processed_dataset = self.py_preprocessing_step.process_step(self.image_dataset)
        for img in processed_dataset.take(1):
            self.assertEqual(img.dtype, tf.float16)

    def test_output_datatype_default(self):
        image_datatype_kept = StepBase.default_output_datatype
        StepBase.default_output_datatype = tf.uint8
        tf_preprocessing_step = TfTestStep(**self.local_vars)
        processed_dataset = tf_preprocessing_step.process_step(self.image_dataset)
        for img in processed_dataset.take(1):
            self.assertEqual(img.dtype, tf.uint8)
        StepBase.default_output_datatype = tf.float16
        tf_preprocessing_step = TfTestStep(**self.local_vars)
        processed_dataset = tf_preprocessing_step.process_step(self.image_dataset)
        for img in processed_dataset.take(1):
            self.assertEqual(img.dtype, tf.float16)
        # Check if 'default_output_datatypes' in 'StepBase' remains unchanged, when Child Class changes 'output_datatypes' attribute.
        tf_preprocessing_step.output_datatype = tf.uint8
        self.assertEqual(StepBase.default_output_datatype, tf.float16)
        StepBase.default_output_datatype = image_datatype_kept 
    
    def test_equal_objects(self):
        self.assertEqual(self.py_preprocessing_step, self.tf_preprocessing_step)

    def test_not_equal_objects(self):
        local_vars = { 'param1': 20, 'param2': (20,20), 'param3': False}
        tf_preprocessing_step = TfTestStep(**local_vars)
        self.assertNotEqual(self.py_preprocessing_step, tf_preprocessing_step)


if __name__ == '__main__':
    unittest.main()
