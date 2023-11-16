import os
import unittest
from unittest.mock import patch

import tensorflow as tf
import cv2

from python_code.image_preprocessing.image_preprocessor import ImagePreprocessor
from python_code.image_preprocessing.preprocessing_steps.step_base import StepBase
from python_code.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape
from python_code.load_raw_data.kaggle_dataset import load_tf_record


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_TEST_PATH = os.path.join(ROOT_DIR, r'python_code/image_preprocessing/configuration/test_image_preprocessor.json')


class TestStepBase(unittest.TestCase):
    """
    A test class derived from unittest.TestCase that verifies the functionality of image preprocessing steps.

    This test suite checks the integrity and flow of a preprocessing pipeline, making conversions between
    grayscale and RGB, ensuring that image shapes are maintained and targets remain unaltered after processing.
    It also verifies the capability to save and load preprocessing pipelines to and from a JSON configuration file.
    
    The suite comprises unit tests for individual preprocessing steps and integration tests for the overall
    preprocessing pipeline with a focus on parameter handling, image data type conversions, and pipeline persistence. Also, 
    these tests ensure that the step classes correctly process image datasets and handle exceptions as expected when 
    integrated in the ImagePreprocessing pipeline
    """

    class GrayscaleToRGB(StepBase):

        arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
        name = 'Grayscale_to_RGB'

        def __init__(self, param1=10 , param2=(10,10), param3=True):
            super().__init__(locals())

        @StepBase._tf_function_decorator
        def process_step(self, tf_image, tf_target):
            tf_image_grayscale = tf.image.grayscale_to_rgb(tf_image)
            tf_image_grayscale = correct_tf_image_shape(tf_image_grayscale)
            return tf_image_grayscale, tf_target

    class RGBToGrayscale(StepBase):
        
        arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
        name = 'RGB_to_Grayscale'
        
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
        
    class ErrorStep(StepBase):

        name = 'ErrorStep'
        
        def __init__(self):
            super().__init__(locals())
            
        @StepBase._py_function_decorator
        def process_step(self, tf_image, tf_target):
            cv_img = (tf_image.numpy()).astype('uint8')
            cv_blurred_image = cv2.GaussianBlur(cv_img, oops_unknown_parameter_here='sorry')  
            tf_blurred_image = tf.convert_to_tensor(cv_blurred_image, dtype=tf.uint8)
            return (tf_blurred_image, tf_target)


    @classmethod
    def setUpClass(cls):
        cls.image_dataset = load_tf_record().take(9)        # To reduce testing time test cases share this attribute. Do not change this attribute!
    
    def setUp(self):
        with open(JSON_TEST_PATH, 'a'): pass
        self.pipeline = [
            TestStepBase.RGBToGrayscale(param1=20,param2=(20,20),param3=False),
            TestStepBase.GrayscaleToRGB(param1=40,param2=(30,30),param3=False),
            TestStepBase.RGBToGrayscale(param1=30,param2=(10,10),param3=True),  
            TestStepBase.GrayscaleToRGB(param1=40,param2=(30,30),param3=False),
            TestStepBase.RGBToGrayscale(param1=30,param2=(10,10),param3=False)  
        ]

    def tearDown(self):
        if os.path.exists(JSON_TEST_PATH):
            os.remove(JSON_TEST_PATH)
    
    def test_process_pipeline(self):
        """    Tests the functionality of the image preprocessing pipeline.

        This test case validates that the pipeline, when applied to an image dataset,
        correctly processes images through multiple preprocessing steps and maintains
        the integrity of the images' shape, specifically ensuring the color channel conversion was done and the 
        dimension is correct after processing.
        """
        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(self.pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)

    def test_save_and_load_pipeline(self):
        """    Ensures the image preprocessing pipeline can be saved and subsequently loaded.

        This test case checks the pipeline's persistence mechanism, verifying that the
        pipeline can be serialized to a JSON configuration and reloaded to create an
        identical pipeline setup.
        """
        
        mock_mapping = {'RGB_to_Grayscale': TestStepBase.RGBToGrayscale, 'Grayscale_to_RGB': TestStepBase.GrayscaleToRGB}
        with patch('python_code.image_preprocessing.image_preprocessor.STEP_CLASS_MAPPING', mock_mapping):
            old_preprocessor = ImagePreprocessor()
            old_preprocessor.set_pipe(self.pipeline)
            old_preprocessor.save_pipe_to_json(JSON_TEST_PATH)
            new_preprocessor = ImagePreprocessor()
            new_preprocessor.load_pipe_from_json(JSON_TEST_PATH)

        self.assertEqual(len(old_preprocessor._pipeline), len(new_preprocessor._pipeline), 'Pipeline lengths are not equal.')
        for old_step, new_step in zip(old_preprocessor._pipeline, new_preprocessor._pipeline):
            self.assertEqual(old_step, new_step, 'Pipeline steps are not equal.')
        
        processed_dataset = new_preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
    
    def test_not_raised_step_process_exception_1(self):
        """   Test case for ensuring that the ErrorStep subclass, when processing an image
        dataset, raises an exception as expected, but the exception is caught and 
        handled silently by the ImagePreprocessor pipeline, allowing the execution
        to continue without interruption.
        """

        pipeline = [
             TestStepBase.RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             TestStepBase.ErrorStep()
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self.assertIsNone(processed_dataset)
    
    def test_not_raised_step_process_exception_2(self):
        """   Test case for ensuring that when pipeline construction is faulty, when processing an image
        dataset, raises an exception as expected, but the exception is caught and 
        handled silently by the ImagePreprocessor pipeline, allowing the execution
        to continue without interruption.
        """

        pipeline = [
             TestStepBase.RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             TestStepBase.RGBToGrayscale(param1=20,param2=(20,20),param3=False)     
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self.assertIsNone(processed_dataset)
    
    def test_pipe_pop_and_push(self):
        """
        Tests the functionality of popping and pushing steps in the image preprocessing pipeline.
        
        This test case first populates the pipeline with specific steps, then pops the last step, 
        and finally pushes it back. It verifies both the popped step and the integrity of the 
        pipeline after these operations.
        """
        pipeline = [
             TestStepBase.RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             TestStepBase.GrayscaleToRGB(param1=40,param2=(30,30),param3=False),     
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        popped_step = preprocessor.pipe_pop()

        self.assertEqual(popped_step, TestStepBase.GrayscaleToRGB(param1=40,param2=(30,30),param3=False))
        self.assertEqual(preprocessor._pipeline, pipeline[:1])

        preprocessor.pipe_push(popped_step)
        self.assertEqual(preprocessor._pipeline, pipeline)

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):

        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1])   # Check if targets are equal.
            self.assertEqual(processed_data[0].shape[:1], original_data[0].shape[:1]) # Check if height and width are equal.
            self.assertEqual(color_channel_expected, processed_data[0].shape[2])     


if __name__ == '__main__':
    unittest.main()
