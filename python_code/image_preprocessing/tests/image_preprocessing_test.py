import os
import json
import unittest
from unittest.mock import patch

import tensorflow as tf
import cv2

from python_code.image_preprocessing.image_preprocessor import ImagePreprocessor
from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.load_raw_data.kaggle_dataset import load_tf_record

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_TEST_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/config/test_image_preprocessor.json')


class TestStepBase(unittest.TestCase):

    class GrayscaleToRGB(StepBase):
        def __init__(self, param1=10 , param2=(10,10), param3=True, set_params_from_range=False):
            super().__init__('Grayscale_to_RGB', locals())

        @StepBase._tf_function_decorator
        def process_step(self, tf_image, tf_target):
            tf_image_grayscale = tf.image.grayscale_to_rgb(tf_image)
            self.correct_shape(tf_image_grayscale)
            return tf_image_grayscale, tf_target

    class RGBToGrayscale(StepBase):
        def __init__(self, param1=10 , param2=(10,10), param3=True, set_params_from_range=False):
            super().__init__('RGB_to_Grayscale', locals())
            
        @StepBase._py_function_decorator
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
        self.local_vars = {'set_params_from_range': False, 'param1': 10, 'param2': (10,10), 'param3': True}
        self.rgb_to_grayscale = self.RGBToGrayscale(**self.local_vars)
        self.grayscale_to_rgb = self.GrayscaleToRGB(**self.local_vars)

    def tearDown(self):
        if os.path.exists(JSON_TEST_PATH):
            os.remove(JSON_TEST_PATH)
    
    def test_process_pipeline(self):
        
        pipeline = [
            self.rgb_to_grayscale,
            self.grayscale_to_rgb,
            self.rgb_to_grayscale
        ]

        preprocessor = ImagePreprocessor(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel=1)

    def test_save_and_load_pipeline(self):

        pipeline = [
            self.rgb_to_grayscale,
            self.grayscale_to_rgb,
            self.rgb_to_grayscale
        ]

        old_preprocessor = ImagePreprocessor(pipeline)
        old_preprocessor.save_pipe_to_json(JSON_TEST_PATH)
        mock_mapping = {'RGB_to_Grayscale': TestStepBase.RGBToGrayscale, 'Grayscale_to_RGB': TestStepBase.GrayscaleToRGB}
        with patch('python_code.image_preprocessing.image_preprocessor.STEP_CLASS_MAPPING', mock_mapping):
            new_preprocessor = ImagePreprocessor()
            new_preprocessor.load_pipe_from_json(JSON_TEST_PATH)

        self.assertEqual(len(old_preprocessor._pipeline), len(new_preprocessor._pipeline), 'Pipeline lengths are not equal.')
        for old_step, new_step in zip(old_preprocessor._pipeline, new_preprocessor._pipeline):
            self.assertEqual(old_step, new_step, 'Pipeline steps are not equal.')

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel):

        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1])   # Check if targets are equal.
            self.assertEqual(processed_data[0].shape[:1], original_data[0].shape[:1]) # Check if height and width are equal.
            self.assertEqual(color_channel, processed_data[0].shape[2])     


if __name__ == '__main__':
    unittest.main()
