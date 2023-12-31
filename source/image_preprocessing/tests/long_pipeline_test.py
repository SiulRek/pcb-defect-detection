""" This module contains tests for long image preprocessing pipelines. 

The task is to build large pipelines (not necessarily the most efficient ones) and 
test them on a dataset. The purpose of these long pipeline tests is to apply a 
preconfigured image preprocessing pipeline to a dataset and verify if any issues or 
bugs occur during the processing. This helps build confidence in the framework's ability 
to handle complex pipelines without any issues."""

import os
import unittest
import random

import tensorflow as tf

from source.image_preprocessing.image_preprocessor import ImagePreprocessor
from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING
from source.load_raw_data.kaggle_dataset import load_tf_record
from source.utils import TestResultLogger, copy_json_exclude_entries, ClassInstancesSerializer


N = 10  # Number of Pipelines Tests to run.
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/image_preprocessing/tests/outputs')
JSON_TEMPLATE_FILE = os.path.join(ROOT_DIR, r'source/image_preprocessing/pipelines/template.json')
JSON_TEST_FILE = os.path.join(OUTPUT_DIR, 'test_pipe.json')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class RGBToGrayscale(StepBase):
        """  A preprocessing step that converts RGB image to Grayscale image."""
        arguments_datatype = {}
        name = 'RGB To Grayscale'

        def __init__(self):
            """ Initializes the RGBToGrayscale object that can be integrated in an image preprocessing pipeline."""
            super().__init__(locals())

        @StepBase._tensor_pyfunc_wrapper
        def process_step(self, image_tensor):
            if image_tensor.shape[2] == 3:
                processed_image = tf.image.rgb_to_grayscale(image_tensor)
                return processed_image
            return image_tensor
        

class TestLongPipeline(unittest.TestCase):
    """
    This class is the base class for all the long pipeline tests. 

    The goal of the long pipeline tests is to identify if issues occure or bugs appear during processing 
    of long pipelines.

    The class creates an image preprocessor with a preconfigured pipeline. The pipeline is constructed from 
    a JSON template that contains all the image preprocessing steps of the framewor (some exclusions). The steps in the 
    pipeline are shuffled randomly. The class includes a method called test_process_pipeline which applies 
    the preprocessor to a dataset of images and verifies the output. If successful, it attempts to convert the processed 
    dataset to grayscale and verifies the conversion as well. If the dataset is accurately converted to grayscale, it can be 
    inferred that the pipeline has processed the dataset without any issues.
    """
    pipeline_id = None

    @classmethod
    def setUpClass(cls):
        
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        
        cls.image_dataset = load_tf_record().take(9)        # To reduce testing time test cases share this attribute Do not change this attribute.
        cls.logger = TestResultLogger(LOG_FILE, f'Long Pipeline Test {cls.pipeline_id}')
    
    def setUp(self):
        with open(JSON_TEST_FILE, 'a'):
            pass
        self.preprocessor = self.create_preconfigured_image_preprocessor()
        #print(self.preprocessor.get_pipe_code_representation())

    def tearDown(self):
        if os.path.exists(JSON_TEST_FILE):
            os.remove(JSON_TEST_FILE)
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def create_preconfigured_image_preprocessor(self):
        excluded_keys  = [
            "Non Local Mean Denoiser", 
            "RGB To Grayscale", 
            "Grayscale To RGB", 
            "Min Max Normalizer", 
            "Standard Normalizer", 
            "Mean Normalizer", 
            "Local Contrast Normalizer",
            "Type Caster"
        ]
        copy_json_exclude_entries(JSON_TEMPLATE_FILE, JSON_TEST_FILE, excluded_keys)

        serializer = ClassInstancesSerializer(STEP_CLASS_MAPPING)
        pipeline = serializer.get_instances_from_json(JSON_TEST_FILE)
        random.shuffle(pipeline)

        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(pipeline)
        return preprocessor

    def _verify_image_shapes(self, processed_dataset, original_dataset, color_channel_expected):
        for original_data, processed_data in zip(original_dataset, processed_dataset):
            if processed_data[1] != original_data[1]:   # Check if targets are equal.
                return False
            if processed_data[0].shape[0] != processed_data[0].shape[1]: # Check if height and width are equal in processed data.
                return False
            if color_channel_expected != processed_data[0].shape[2]:
                return False
        return True

    def test_process_pipeline(self):
        
        try:
            processed_dataset = self.preprocessor.process(self.image_dataset)
        except Exception as e:
            raise BrokenPipeError('An exception occured while processing the dataset. This is the problematic pipeline: \n'
                       + self.preprocessor.get_pipe_code_representation()) from e
        
        if not self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=3):
            message = 'The processed dataset has unexpected shapes. This is the problematic pipeline: \n' + self.preprocessor.get_pipe_code_representation()
            self.fail(message)
        
        grayscaled_dataset = RGBToGrayscale().process_step(processed_dataset)
        if not self._verify_image_shapes(grayscaled_dataset, self.image_dataset, color_channel_expected=1):
            message = 'The processed dataset could not be converted to grayscale correctly. This is the problematic pipeline: \n' + self.preprocessor.get_pipe_code_representation()
            self.fail(message)


def load_long_pipeline_tests(n=N):
    """
    This function dynamically creates n classes inheriting from TestLongPipeline
    and load them into a test suite. 

    Args:
    - n (int): Number of TestLongPipeline classes to create. Default to N.
    """

    test_suites = []
    for i in range(1, N + 1):
        test_class_name = f'{TestLongPipeline.__name__}{i}'
        test_class = type(test_class_name, (TestLongPipeline,), {'pipeline_id': i})
        test_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suites.append(test_suite)

    test_suite = unittest.TestSuite(test_suites)  # Combine the suites
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_long_pipeline_tests(N))
