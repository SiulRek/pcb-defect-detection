# Image Preprocessing Test Framework
This folder contains test suites and modules for evaluating the components of the image preprocessing framework, ensuring the correct functionality and reliability of these components.

## Modules Overview

[image_preprocessing_test.py](image_preprocessing_test.py): This module is a test suite for the `ImagePreprocessor` class, focusing on the robustness and reliability of image preprocessing operations under various conditions. It includes tests for adding/removing steps, pipeline execution, exception handling, serialization/deserialization, and maintaining image integrity through diverse preprocessing stages.

[step_base_test.py](./step_base_test.py): This test suite tests the StepBase class in an image preprocessing module, ensuring adequate TensorFlow and Python-based preprocessing step initialization and operation. It focuses on image shape transformation, object equality, JSON representation, wrapper function efficacy, and datatype handling.

[single_step_test.py](./single_step_test.py): This test suite tests individual image preprocessing steps to be integrated in the image processing framework. The class focuses on ensuring the correct 
    functioning of the step under test, both in isolation and when integrated into a pipeline. 

[multiple_steps_test.py](./multiple_steps_test.py): This module generates multiple test suites for various image preprocessing steps dynamically, with each class inheriting from `TestSingleStep`. It makes use of Python's dynamic class creation capabilities and allows for selective test execution based on configuration flags.

[multiple_steps_meta_test.py](./multiple_steps_meta_test.py): This module contains tests for validating the functionality of the functionality of [multiple_steps_test.py](./multiple_steps_test.py) that focuses on dynamically generated test classes. It provides tests to ensure that test classes are correctly created for one example preprocessing step and that conditional test logic (skipping certain tests) functions as expected.

[resize_operations_steps_test.py](./resize_operations_steps_test.py): Costumized test suites, inheriting from `TestSingleStep`, for preprocessing steps that perform image resize operations.

[channel_conversions_steps_test.py](./channel_conversions_steps_test.py): Costumized test suites, inheriting from `TestSingleStep`, for preprocessing steps converting images between RGB and Grayscale channel formats.

[test_runner.py](./test_runner.py): Runs most of the test cases from the image preprocessing testing framework, combining different test modules.
