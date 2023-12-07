import json
import os
import unittest
from unittest.mock import patch

import tensorflow as tf
import cv2

from source.image_preprocessing.image_preprocessor import ImagePreprocessor
from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_image_tensor_shape
from source.load_raw_data.kaggle_dataset import load_tf_record
from source.utils import TestResultLogger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_TEST_FILE = os.path.join(ROOT_DIR, r'source/image_preprocessing/pipelines/test_pipe.json')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/image_preprocessing/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class GrayscaleToRGB(StepBase):

    arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
    name = 'Grayscale_to_RGB'

    def __init__(self, param1=10 , param2=(10,10), param3=True):
        super().__init__(locals())

    @StepBase._tensor_pyfunc_wrapper
    def process_step(self, image_tensor):
        image_rgb_tensor = tf.image.grayscale_to_rgb(image_tensor)
        image_rgb_tensor = correct_image_tensor_shape(image_rgb_tensor)
        return image_rgb_tensor


class RGBToGrayscale(StepBase):
    
    arguments_datatype = {'param1': int, 'param2':(int,int), 'param3':bool}
    name = 'RGB_to_Grayscale'
    
    def __init__(self, param1=10 , param2=(10,10), param3=True):
        super().__init__(locals())
        
    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        blurred_image = cv2.GaussianBlur(image_nparray, ksize=(5,5), sigmaX=2)  # Randomly choosen action.
        blurred_image = tf.convert_to_tensor(blurred_image, dtype=tf.uint8)
        image_grayscale_tensor = tf.image.rgb_to_grayscale(blurred_image)
        image_grayscale_tensor = correct_image_tensor_shape(image_grayscale_tensor)
        processed_image_nparray = (image_grayscale_tensor.numpy()).astype('uint8')
        return processed_image_nparray
    # Note in real usage conversion of np.array to tensor and viceversa in one process_step is not recommended.

class ErrorStep(StepBase):

    name = 'ErrorStep'
    
    def __init__(self):
        super().__init__(locals())
        
    @StepBase._nparray_pyfunc_wrapper
    def process_step(self, image_nparray):
        processed_image = cv2.GaussianBlur(image_nparray, oops_unknown_parameter_here='sorry')  
        return (processed_image)


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

    @classmethod
    def setUpClass(cls):
        cls.image_dataset = load_tf_record().take(9)        # To reduce testing time test cases share this attribute Do not change this attribute.
        cls.logger = TestResultLogger(LOG_FILE, 'Image Preprocessor Test')
    
    def setUp(self):
        with open(JSON_TEST_FILE, 'a'): pass
        self.pipeline = [
            RGBToGrayscale(param1=20,param2=(20,20),param3=False),
            GrayscaleToRGB(param1=40,param2=(30,30),param3=False),
            RGBToGrayscale(param1=30,param2=(10,10),param3=True),  
            GrayscaleToRGB(param1=40,param2=(30,30),param3=False),
            RGBToGrayscale(param1=30,param2=(10,10),param3=False)  
        ]

    def tearDown(self):
        if os.path.exists(JSON_TEST_FILE):
            os.remove(JSON_TEST_FILE)
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):

        for original_data, processed_data in zip(original_dataset, processed_dataset):
            self.assertEqual(processed_data[1], original_data[1])   # Check if targets are equal.
            self.assertEqual(processed_data[0].shape[:1], original_data[0].shape[:1]) # Check if height and width are equal.
            self.assertEqual(color_channel_expected, processed_data[0].shape[2])     
    
    def test_pipe_pop_and_append(self):
        """
        Tests the functionality of popping and appending steps in the image preprocessing pipeline.
        
        This test case first populates the pipeline with specific steps, then pops the last step, 
        and finally appendes it back. It verifies both the popped step and the integrity of the 
        pipeline after these operations.
        """
        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             GrayscaleToRGB(param1=40,param2=(30,30),param3=False),     
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        popped_step = preprocessor.pipe_pop()

        self.assertEqual(popped_step, GrayscaleToRGB(param1=40,param2=(30,30),param3=False))
        self.assertEqual(preprocessor.pipeline, pipeline[:1])

        preprocessor.pipe_append(popped_step)
        self.assertEqual(preprocessor.pipeline, pipeline)

    def test_pipeline_clear(self):
        """
        Tests the functionality of clearing and reinitializing the image preprocessing pipeline.

        This test case verifies that the `pipe_clear` method of the ImagePreprocessor class effectively clears the 
        existing pipeline and allows to rebuild the pipeline from start. 
        """
        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             GrayscaleToRGB(param1=40,param2=(30,30),param3=False),     
        ]
        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(pipeline)
        preprocessor.pipe_clear()
        self.assertEqual(preprocessor.pipeline, [])
        preprocessor.pipe_append(pipeline[0])
        preprocessor.pipe_append(pipeline[1])
        self.assertEqual(preprocessor.pipeline, pipeline)
        preprocessor.pipe_clear()
        self.assertEqual(preprocessor.pipeline, [])
        preprocessor.set_pipe(pipeline)
        self.assertEqual(preprocessor.pipeline, pipeline)

    def test_deepcopy_of_pipeline(self):
        """
        This test ensures that the ImagePreprocessor maintains a consistent and isolated state of its preprocessing pipeline.

        Assert is equal implies, that the internal pipeline was successfully deep-copied. 
        """
        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             GrayscaleToRGB(param1=40,param2=(30,30),param3=False),     
        ]
        pipeline_expected = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             GrayscaleToRGB(param1=40,param2=(30,30),param3=False),     
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        pipeline.append('Changing pipeline by appending invalid element to it.')
        self.assertEqual(preprocessor.pipeline, pipeline_expected)

    def test_invalid_step_in_pipeline(self):
        """    Tests the ImagePreprocessor's ability to validate the types of steps added to its pipeline.

        This test ensures that the ImagePreprocessor class correctly identifies and rejects any objects 
        added to its pipeline that are not a subclass of StepBase. 
        """
        class StepNotOfStepBase: pass
        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             StepNotOfStepBase,     
        ]
        with self.assertRaises(ValueError):
            preprocessor = ImagePreprocessor()
            preprocessor.set_pipe(pipeline)

    def _remove_new_lines_and_spaces(self, string):
        string = string.replace('\n','')
        string = string.replace(' ','')
        return string

    def test_pipeline_code_representation(self):
        """
        Tests ensures the correctness of the pipeline code representation generated by the ImagePreprocessor.
        """
        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             GrayscaleToRGB(param1=40,param2=(30,30),param3='a nice str'),     
        ]
        representation_expected = """[
        RGBToGrayscale(param1=20,param2=(20,20),param3=False),
        GrayscaleToRGB(param1=40, param2=(30,30),param3='a nice str')
        ]"""
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        representation_output = preprocessor.get_pipe_code_representation()
        representation_output = self._remove_new_lines_and_spaces(representation_output)
        representation_expected = self._remove_new_lines_and_spaces(representation_expected)
        self.assertEqual(representation_output, representation_expected)
    
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
        pipeline can be serialized to JSON and reloaded to create an
        identical pipeline setup.
        """
        
        mock_mapping = {'RGB_to_Grayscale': RGBToGrayscale, 'Grayscale_to_RGB': GrayscaleToRGB}
        with patch('source.image_preprocessing.image_preprocessor.STEP_CLASS_MAPPING', mock_mapping):
            old_preprocessor = ImagePreprocessor()
            old_preprocessor.set_pipe(self.pipeline)
            old_preprocessor.save_pipe_to_json(JSON_TEST_FILE)
            new_preprocessor = ImagePreprocessor()
            new_preprocessor.load_pipe_from_json(JSON_TEST_FILE)

        self.assertEqual(len(old_preprocessor.pipeline), len(new_preprocessor.pipeline), 'Pipeline lengths are not equal.')
        for old_step, new_step in zip(old_preprocessor.pipeline, new_preprocessor.pipeline):
            self.assertEqual(old_step, new_step, 'Pipeline steps are not equal.')
        
        processed_dataset = new_preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
    
    def test_load_randomized_pipeline(self):
        mock_class_parameters = {
            'param1': {'distribution': 'uniform', 'low': 2, 'high': 10}, 
            'param2':'[(3,3)]*10 + [(5,5)]*10 + [(8,8)]', 
            'param3': [True, True]
            }  
        json_data = {'RGB_to_Grayscale': mock_class_parameters}
        
        with open(JSON_TEST_FILE, 'w') as file:
            json.dump(json_data, file)

        mock_mapping = {'RGB_to_Grayscale': RGBToGrayscale}
        with patch('source.image_preprocessing.image_preprocessor.STEP_CLASS_MAPPING', mock_mapping):
            preprocessor = ImagePreprocessor()
            preprocessor.load_randomized_pipe_from_json(JSON_TEST_FILE)
            pipeline = preprocessor.pipeline

        self.assertIsInstance(pipeline[0], RGBToGrayscale)
        self.assertTrue(2 <= pipeline[0].parameters['param1'] <= 10)
        self.assertIn(pipeline[0].parameters['param2'], [(3,3),(5,5),(8,8)])
        self.assertTrue(pipeline[0].parameters['param3'])
        # TODO here
    
    def test_not_raised_step_process_exception_1(self):
        """   Test case for ensuring that the ErrorStep subclass, when processing an image
        dataset, raises an exception as expected, but the exception is caught and 
        handled silently by the ImagePreprocessor pipeline, allowing the execution
        to continue without interruption.
        """

        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             ErrorStep()
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self.assertIn("missing required argument 'ksize'",preprocessor.occurred_exception_message)
        self.assertIsNone(processed_dataset)
    
    def test_not_raised_step_process_exception_2(self):
        """   Test case for ensuring that when pipeline construction is faulty, when processing an image
        dataset, raises an exception as expected, but the exception is caught and 
        handled silently by the ImagePreprocessor pipeline, allowing the execution
        to continue without interruption.
        """

        pipeline = [
             RGBToGrayscale(param1=20,param2=(20,20),param3=False),
             RGBToGrayscale(param1=20,param2=(20,20),param3=False)     
        ]
        preprocessor = ImagePreprocessor(raise_step_process_exception=False)
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self.assertIn("An error occurred in step RGB_to_Grayscale",preprocessor.occurred_exception_message)
        self.assertIsNone(processed_dataset)


if __name__ == '__main__':
    unittest.main()
