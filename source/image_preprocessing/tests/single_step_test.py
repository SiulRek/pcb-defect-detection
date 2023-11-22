"""
This module contains a suite of tests designed to validate image preprocessing steps before their integration into an image preprocessing pipeline. Each preprocessing step must successfully pass all the tests specified in this module to ensure its functionality, compatibility, and reliability within the pipeline!
"""

import json
import os
import unittest
from unittest.mock import patch

import tensorflow as tf

#TODO Select Step to test here!
from source.image_preprocessing.preprocessing_steps import NLMeanDenoiser as StepToTest

from source.image_preprocessing.image_preprocessor import ImagePreprocessor
from source.image_preprocessing.preprocessing_steps.step_base import StepBase
from source.image_preprocessing.preprocessing_steps.step_utils import correct_tf_image_shape
from source.image_preprocessing.preprocessing_steps.step_class_mapping import STEP_CLASS_MAPPING
from source.load_raw_data.kaggle_dataset import load_tf_record
from source.utils import recursive_type_conversion,  PCBVisualizerforTF
from source.utils import SimplePopupHandler, TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
JSON_TEST_FILE = os.path.join(ROOT_DIR, r'source/image_preprocessing/pipelines/test_pipe.json')
JSON_TEMP_FILE = os.path.join(ROOT_DIR, r'source/image_preprocessing/pipelines/template.json')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/image_preprocessing/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class RGBToGrayscale(StepBase):
    arguments_datatype = {}
    name = 'RGB_to_Grayscale'

    def __init__(self):
        super().__init__(locals())

    @StepBase._tf_function_decorator
    def process_step(self, image_tensor):
        image_grayscale_tensor = tf.image.rgb_to_grayscale(image_tensor)
        image_grayscale_tensor = correct_tf_image_shape(image_grayscale_tensor)
        return image_grayscale_tensor


class GrayscaleToRGB(StepBase):
    arguments_datatype = {}
    name = 'Grayscale_to_RGB'

    def __init__(self):
        super().__init__(locals())

    @StepBase._tf_function_decorator
    def process_step(self, image_tensor):
        image_tensor = tf.image.grayscale_to_rgb(image_tensor)
        image_grayscale_tensor = correct_tf_image_shape(image_tensor)
        return image_grayscale_tensor


class TestSingleStep(unittest.TestCase):
    """
    A unit test class for testing individual image preprocessing steps in a 
    TensorFlow image processing pipeline. The class focuses on ensuring the correct 
    functioning of these steps, both in isolation and when integrated into a pipeline. 
    The tests cover: processing of RGB and grayscale images, 
    saving and loading of the pipeline pipelines, and validation of instance 
    argument data types.
    """

    # Class Attributes (overwritten when class is dynamically loaded -> multiple_steps_test.py)
    params = {'h': 1.0, 'template_window_size': 7, 'search_window_size': 21}
    StepClass = StepToTest
    process_grayscale_only = False

    @classmethod
    def setUpClass(cls):

        cls.image_dataset = load_tf_record().take(9)        
        
        cls.popup_handler = SimplePopupHandler()
        cls.logger = TestResultLogger(LOG_FILE, f'{StepToTest.name} Test')

        step_name_edit = cls.StepClass.name.replace(' ', '_').lower()
        cls.step_output_dir = os.path.join(OUTPUT_DIR, step_name_edit)
        if __name__ == '__main__':
            cls.visual_inspection = cls.popup_handler.ask_yes_no_question('Do you want to make a visual inspection?')

        if not os.path.isdir(cls.step_output_dir):
            os.makedirs(cls.step_output_dir)
                
    def setUp(self):
        with open(JSON_TEST_FILE, 'a'): pass
        self.test_step = self.StepClass(**self.params)

    def tearDown(self):
        if os.path.exists(JSON_TEST_FILE):
            os.remove(JSON_TEST_FILE)   
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

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

    def test_arguments_datatype(self):
        """ 
        Test to verify that the datatype specifications for StepToTest instance parameters are correct.
        Ensures that the actual parameters match the expected datatypes specified in the class.
        """

        params = self.test_step.params
        init_params_datatype = self.StepClass.arguments_datatype
        
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
        self.assertIs(STEP_CLASS_MAPPING[step_name], self.StepClass, 'Mapped value of tested step is not the tested step class itself.')
    
    def test_process_rgb_images(self):
        """ 
        Test to ensure that RGB images are processed correctly. 
        Verifies that the RGB images, after processing, have the expected color channel dimensions.
        """
        pipeline = [self.test_step, RGBToGrayscale()]
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
        pipeline = [RGBToGrayscale(), self.test_step]
        preprocessor = ImagePreprocessor()
        preprocessor.set_pipe(pipeline)
        processed_dataset = preprocessor.process(self.image_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=1)
        processed_dataset = GrayscaleToRGB().process_step(processed_dataset)
        self._verify_image_shapes(processed_dataset, self.image_dataset, color_channel_expected=3)
    
    def test_load_from_json(self):
        """ This method tests the functionality of loading a preprocessing step from a JSON File. It verifies that the specified preprocessing step, StepToTest, is correctly instantiated and configured based on the settings provided in the JSON file. This ensures the JSON Files's compatibility and correctness with the pipeline instantiation process.
        """

        step_name = self.test_step.name

        with open(JSON_TEMP_FILE, 'r') as file:
            json_data = json.load(file)

        self.assertIn(step_name, json_data.keys(), 'StepToTest has no entry in JSON template.')
        
        preprocessor = ImagePreprocessor()
        preprocessor.load_pipe_from_json(JSON_TEMP_FILE)

        step_is_instance = [isinstance(step, self.StepClass) for step in preprocessor.pipeline]
        self.assertIn(True, step_is_instance)

    def test_save_and_load_pipeline(self):
        """ 
        Test to ensure the functionality of saving and loading the preprocessing pipeline.
        Confirms that the pipeline configuration is correctly preserved across save and load operations.
        """
        
        mock_mapping = {'RGB_to_Grayscale': RGBToGrayscale, 'Test_Step': self.StepClass}
        with patch('source.image_preprocessing.image_preprocessor.STEP_CLASS_MAPPING', mock_mapping):
            old_preprocessor = ImagePreprocessor()
            pipeline = [RGBToGrayscale(), self.test_step]
            old_preprocessor.set_pipe(pipeline)
            old_preprocessor.save_pipe_to_json(JSON_TEST_FILE)
            new_preprocessor = ImagePreprocessor()
            new_preprocessor.load_pipe_from_json(JSON_TEST_FILE)

        self.assertEqual(len(old_preprocessor._pipeline), len(new_preprocessor._pipeline), 'Pipeline lengths are not equal.')
        for old_step, new_step in zip(old_preprocessor._pipeline, new_preprocessor._pipeline):
            self.assertEqual(old_step, new_step, 'Pipeline steps are not equal.')
    
    #@unittest.skip("Visual inspection not enabled")
    def test_processed_image_visualization(self):
        """ This method evaluates the visualization capabilities for processed images. It processes RGB and grayscale images through the StepToTest, visualizes them using PCBVisualizerforTF, and saves these visualizations to files. The method allows processed images to be visually inspected."""
        
        pcb_visualizer = PCBVisualizerforTF(show_plot=False)
        if not self.process_grayscale_only:
            processed_rgb_dataset = self.test_step.process_step(self.image_dataset) 
            pcb_visualizer.plot_images(processed_rgb_dataset, 'Processed RGB Images')
            figure_name = 'processed_rgb_images'
            pcb_visualizer.save_plot_to_file(os.path.join(self.step_output_dir, figure_name)) 
            pcb_visualizer.plot_image_comparison(self.image_dataset, processed_rgb_dataset, 1,'RGB Images comparison')
            figure_name = 'rgb_images_comparison'
            pcb_visualizer.save_plot_to_file(os.path.join(self.step_output_dir, figure_name))

        grayscaled_dataset = RGBToGrayscale().process_step(self.image_dataset) 
        processed_grayscaled_dataset = self.test_step.process_step(grayscaled_dataset) 
        pcb_visualizer.plot_images(processed_grayscaled_dataset, 'Processed Grayscale Images')
        figure_name = 'processed_grayscaled_images'
        pcb_visualizer.save_plot_to_file(os.path.join(self.step_output_dir, figure_name))
        pcb_visualizer.plot_image_comparison(grayscaled_dataset, processed_grayscaled_dataset, 1,'Grayscale Images comparison')
        figure_name = 'grayscaled_images_comparison'
        pcb_visualizer.save_plot_to_file(os.path.join(self.step_output_dir, figure_name))
    

if __name__ == '__main__':
    unittest.main()
