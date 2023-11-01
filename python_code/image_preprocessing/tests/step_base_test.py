import unittest
import os
import json

import tensorflow as tf
import cv2

from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.load_raw_data.kaggle_dataset import load_tf_record


class TestStepBase(unittest.TestCase):

    class TfTestStep(StepBase):
        def __init__(self, param1=10 , param2=(10,10), param3=True, set_random_params=False):
            super().__init__('Test_Step', locals())

        @StepBase.tf_function_decorator
        def process_step(self, tf_image, tf_target):
            tf_image_grayscale = tf.image.rgb_to_grayscale(tf_image)
            self.correct_shape(tf_image_grayscale)
            return tf_image_grayscale, tf_target

    class PyTestStep(StepBase):
        def __init__(self, param1=10 , param2=(10,10), param3=True, set_random_params=False):
            super().__init__('Test_Step', locals())

        @StepBase.py_function_decorator
        def process_step(self, tf_image, tf_target):
            cv_img = (tf_image.numpy()).astype('uint8')
            cv_blurred_image = cv2.GaussianBlur(cv_img, ksize=(5,5), sigmaX=2)  # Randomly choosen action.
            tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
            tf_image_grayscale = tf.image.rgb_to_grayscale(tf_blurred_image)
            tf_blurred_image = self.correct_shape(tf_blurred_image)
            return (tf_image_grayscale, tf_target)

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_dataset = load_tf_record().take(9)
    
    def setUp(self):
        self.local_vars = {'set_random_params': False, 'param1': 10, 'param2': (10,10), 'param3': True}
        self.tf_preprocessing_step = self.TfTestStep(**self.local_vars)
        self.py_preprocessing_step = self.PyTestStep(**self.local_vars)
    
    def test_initialization(self):
        self.assertEqual(self.tf_preprocessing_step.name, "Test_Step")
        self.assertEqual(self.tf_preprocessing_step.params, {'param1': 10, 'param2': (10,10), 'param3': True})
        
    def test_load_params_from_json(self):
        configs = self.tf_preprocessing_step.load_params_from_json()
        self.assertIsInstance(configs, dict)

    def test_random_params(self):
        self.local_vars['set_random_params'] = True
        self.tf_preprocessing_step.random_params()
        self.assertIn(self.tf_preprocessing_step.params['param1'], [10,20,30,40])  # Adjust based on your json file
        self.assertIn(self.tf_preprocessing_step.params['param2'], [(10,10),(20,20),(30,30)])  # Adjust based on your json file
        self.assertIn(self.tf_preprocessing_step.params['param3'], [True, False])  # Adjust based on your json file

    def test_correct_shape_gray(self):
        tf_image = [x for x in TestStepBase.image_dataset.take(1)][0][0]
        tf_image_grayscale = tf.image.rgb_to_grayscale(tf_image)
        reshaped_image = self.tf_preprocessing_step.correct_shape(tf_image_grayscale)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 1])

    def test_correct_shape_rgb(self):
        tf_image = [x for x in TestStepBase.image_dataset.take(1)][0][0]
        reshaped_image = self.tf_preprocessing_step.correct_shape(tf_image)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 3])

    def test_tf_function_decorator(self):
        tf_dataset = self.tf_preprocessing_step.process_step(self.image_dataset)
        tf_image = [x for x in tf_dataset.take(1)][0][0]
        reshaped_image = self.tf_preprocessing_step.correct_shape(tf_image)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 1])

    def test_py_function_decorator(self):
        tf_dataset = self.py_preprocessing_step.process_step(self.image_dataset)
        tf_image = [x for x in tf_dataset.take(1)][0][0]
        reshaped_image = self.tf_preprocessing_step.correct_shape(tf_image)
        self.assertEqual(reshaped_image.shape, [2464, 3056, 1])
        self.assertEqual(tf_image.shape, [2464, 3056, 1])


if __name__ == '__main__':
    unittest.main()
