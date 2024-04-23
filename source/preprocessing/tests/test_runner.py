import unittest
import os
from unittest import defaultTestLoader as Loader

from source.preprocessing.tests.for_preprocessor import image_preprocessing_test
from source.preprocessing.tests.for_preprocessor.long_pipeline_test import (
    load_long_pipeline_tests,
)
from source.preprocessing.tests.for_steps.multiple_steps_test import (
    load_multiple_steps_tests,
)
from source.preprocessing.tests.for_steps.channel_conversions_steps_test import (
    load_channel_conversion_steps_tests,
)
from source.preprocessing.tests.for_steps.resize_operations_steps_test import (
    load_resize_operations_steps_tests,
)
from source.preprocessing.tests.for_steps.data_augmentation_steps_test import (
    load_data_augmentation_steps_tests,
)
from source.utils import TestResultLogger
from source.preprocessing.tests.for_helpers import step_base_test
from source.preprocessing.tests.for_helpers import copy_json_exclude_entries_test
from source.preprocessing.tests.for_helpers import recursive_type_conversion_test
from source.preprocessing.tests.for_helpers import randomly_select_sequential_keys_test
from source.preprocessing.tests.for_helpers import parse_and_repeat_test
from source.preprocessing.tests.for_helpers import class_instance_serializer_test


ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."
)
OUTPUT_DIR = os.path.join(ROOT_DIR, r"source/preprocessing/tests/outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_results.log")


def load_tests(test_suite):
    """
    Populates the given test suite with a series of test cases from the image preprocessing testing
    framework.

    This function aggregates tests from different aspects of the image preprocessing pipeline. It
    includes basic tests, tests for multiple steps, specific channel conversion tests, and general
    image preprocessing tests.

    Args:
        test_suite (unittest.TestSuite): The test suite to which the tests will be added.

    Returns:
        unittest.TestSuite: The test suite populated with a range of tests from different modules.
    """
    test_suite.addTest(Loader.loadTestsFromModule(copy_json_exclude_entries_test))
    test_suite.addTest(Loader.loadTestsFromModule(recursive_type_conversion_test))
    test_suite.addTest(Loader.loadTestsFromModule(randomly_select_sequential_keys_test))
    test_suite.addTest(Loader.loadTestsFromModule(class_instance_serializer_test))
    test_suite.addTest(Loader.loadTestsFromModule(parse_and_repeat_test))
    test_suite.addTest(Loader.loadTestsFromModule(step_base_test))
    test_suite.addTest(Loader.loadTestsFromModule(image_preprocessing_test))
    test_suite.addTest(load_multiple_steps_tests())
    test_suite.addTest(load_channel_conversion_steps_tests())
    test_suite.addTest(load_resize_operations_steps_tests())
    test_suite.addTest(load_data_augmentation_steps_tests())
    test_suite.addTest(load_long_pipeline_tests(1))
    return test_suite


if __name__ == "__main__":
    """Main execution block for running the aggregated test suite."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_result_logger = TestResultLogger(LOG_FILE)  # Initialize Test Result Logger.
    test_suite = unittest.TestSuite()
    unittest.TextTestRunner().run(load_tests(test_suite))
