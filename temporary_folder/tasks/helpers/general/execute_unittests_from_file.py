import importlib
import io
import os
import sys
import unittest


def load_unittests_from_file(file_path):
    """
    Load unittests from a file and return them.

    Args:
        - file_path (str): Path to the file containing the unittests.

    Returns:
        - unittest.TestSuite: The unittests loaded from the file.
    """
    test_file_path = os.path.abspath(file_path)

    test_dir = os.path.dirname(test_file_path)
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)

    module_name = os.path.splitext(os.path.basename(test_file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, test_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return unittest.defaultTestLoader.loadTestsFromModule(module)


def execute_unittests_from_file(file_path, verbosity=1):
    """
    Execute unittests from a file and return the output.

    Args:
        - file_path (str): Path to the file containing the unittests.
        - verbosity (int): Verbosity level of the unittests. Default is 1.

    Returns:
        - str: The output of the unittests.
    """
    tests = load_unittests_from_file(file_path)
    test_output = io.StringIO()
    runner = unittest.TextTestRunner(stream=test_output, verbosity=verbosity)
    runner.run(tests)
    output = test_output.getvalue()
    test_output.close()
    return output


if __name__ == "__main__":
    path = f"path/to/file"
    test_results = execute_unittests_from_file(path)
    print(test_results)
