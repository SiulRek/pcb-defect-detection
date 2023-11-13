"""
This module contains a suite of tests designed to validate image preprocessing steps before their integration into an image preprocessing pipeline. Each preprocessing step must successfully pass all the tests specified in this module to ensure its functionality, compatibility, and reliability within the pipeline!
"""


import os
import unittest
from unittest.mock import patch

import tensorflow as tf
import cv2

# Select Step to test here!
from python_code.image_preprocessing.preprocessing_steps import AdaptiveHistogramEqualization as TestStep

from python_code.image_preprocessing.image_preprocessor import ImagePreprocessor
from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape
from python_code.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING
from python_code.load_raw_data.kaggle_dataset import load_tf_record
from python_code.utils import recursive_type_conversion


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_TEST_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/config/test_image_preprocessor.json')


class TestSingleStep(unittest.TestCase):
    """
    A unit test class for testing individual image preprocessing steps in a 
    TensorFlow image processing pipeline. The class focuses on ensuring the correct 
    functioning of these steps, both in isolation and when integrated into a pipeline. 
    The tests cover: processing of RGB and grayscale images, 
    saving and loading of the pipeline configuration, and validation of instance 
    argument data types.
    """

    class RGBToGrayscale(StepBase):
        _init_params_datatypes = {}

        def __init__(self):
            super().__init__('RGB_to_Grayscale', locals())

        @StepBase._tf_function_decorator
        def process_step(self, tf_image, tf_target):
            tf_image_grayscale = tf.image.rgb_to_grayscale(tf_image)
            tf_image_grayscale = correct_tf_image_shape(tf_image_grayscale)
            return tf_image_grayscale, tf_target
        
    class GrayscaleToRGB(StepBase):
        _init_params_datatypes = {}

        def __init__(self):
            super().__init__('Grayscale_to_RGB', locals())

        @StepBase._tf_function_decorator
        def process_step(self, tf_image, tf_target):
            tf_image_grayscale = tf.image.grayscale_to_rgb(tf_image)
            tf_image_grayscale = correct_tf_image_shape(tf_image_grayscale)
            return tf_image_grayscale, tf_target


    @classmethod
    def setUpClass(cls):
        cls.image_dataset = load_tf_record().take(9)        # To reduce testing time test cases share this attribute. Do not change this attribute!
    
    def setUp(self):
        with open(JSON_TEST_PATH, 'a'): pass
        # TestStep._init_params_datatypes
        self.params = {'clip_limit': 1.2, 'tile_gridsize': (8, 8)}
        self.test_step = TestStep(**self.params)

    def tearDown(self):
        if os.path.exists(JSON_TEST_PATH):
            os.remove(JSON_TEST_PATH)
    
    def test_process_rgb_images(self):
        """ 
        Test to ensure that RGB images are processed correctly. 
        Verifies that the RGB images, after processing, have the expected color channel dimensions.
        """

        pipeline = [self.test_step, TestSingleStep.RGBToGrayscale()]
        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
  
    def test_process_grayscaled_images(self):
        """ 
        Test to ensure that grayscale images are processed correctly.
        Checks if the grayscale images maintain their dimensions after processing and 
        verifies the color channel transformation correctness.
        """

        pipeline = [TestSingleStep.RGBToGrayscale(), self.test_step]
        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
        processed_dataset = TestSingleStep.GrayscaleToRGB().process_step(processed_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=3)

    def test_save_and_load_pipeline(self):
        """ 
        Test to ensure the functionality of saving and loading the preprocessing pipeline.
        Confirms that the pipeline configuration is correctly preserved across save and load operations.
        """
        
        mock_mapping = {'RGB_to_Grayscale': TestSingleStep.RGBToGrayscale, 'Test_Step': TestStep}
        with patch('python_code.image_preprocessing.image_preprocessor.STEP_CLASS_MAPPING', mock_mapping):
            old_preprocessor = ImagePreprocessor()
            pipeline = [TestSingleStep.RGBToGrayscale(), self.test_step]
            old_preprocessor.set_pipe(pipeline)
            old_preprocessor.save_pipe_to_json(JSON_TEST_PATH)
            new_preprocessor = ImagePreprocessor()
            new_preprocessor.load_pipe_from_json(JSON_TEST_PATH)

        self.assertEqual(len(old_preprocessor._pipeline), len(new_preprocessor._pipeline), 'Pipeline lengths are not equal.')
        for old_step, new_step in zip(old_preprocessor._pipeline, new_preprocessor._pipeline):
            self.assertEqual(old_step, new_step, 'Pipeline steps are not equal.')

    def test_init_params_datatypes(self):
        """ 
        Test to verify that the datatype specifications for TestStep instance parameters are correct.
        Ensures that the actual parameters match the expected datatypes specified in the class.
        """

        params = self.test_step.params
        init_params_datatype = TestStep._init_params_datatypes
        
        self.assertEqual(params.keys(), init_params_datatype.keys(), "'init_params_datatype' keys does not match with 'params' attribute.")

        for key in params.keys():
            param_converted = recursive_type_conversion(params[key], init_params_datatype[key]) # When everything goes right, no conversion is done.
            self.assertEqual(param_converted, params[key], "'init_params_datatype' specification is incorrect.")
    
    def test_mapping_entry_of_step(self):
        """ 
        Test to verify the presence and correctness of the mapping entry for the tested preprocessing step.
        Checks if the step class is correctly mapped and if the mapping points to the step itself.
        """
        step_name = self.test_step.name
        self.assertIn(step_name, STEP_CLASS_MAPPING.keys(), 'No mapping is specified for the tested step.')
        self.assertIs(STEP_CLASS_MAPPING[step_name], TestStep, 'Mapped value of tested step is not the tested step class itself.')

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):
        """ 
        Helper method to verify the image dimensions and color channels in a processed dataset.
        Compares the processed images to the original dataset to ensure correct height, width, 
        and color channel transformations.
        """
        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1], 'Targets are not equal.')  
            self.assertEqual(processed_data[0].shape[:1], original_data[0].shape[:1], 'height and/or width are not equal.') 
            self.assertEqual(color_channel_expected, processed_data[0].shape[2], 'Color channels are not equal.')     
    


if __name__ == '__main__':
    unittest.main()
