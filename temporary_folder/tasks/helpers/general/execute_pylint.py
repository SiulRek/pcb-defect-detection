import os
import subprocess
import sys


def execute_pylint(python_file_path, environment_path):
    """
    Runs pylint on the specified Python file and prints the output.

    Args:
        - python_file_path (str): The path to the Python file to lint.
        - environment_path (str): The path to the Python environment to use.

    Returns:
        - str: The output of the pylint run.
    """
    if sys.platform == "win32":
        pylint_path = os.path.join(environment_path, "Scripts", "pylint")
    else:
        pylint_path = os.path.join(environment_path, "bin", "pylint")
    try:
        result = subprocess.run(
            [pylint_path, python_file_path], capture_output=True, text=True
        )

        if result.stderr:
            print("Errors encountered during pylint run:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        return result.stdout
    except Exception as e:
        print(f"Failed to run pylint on {python_file_path}: {e}", file=sys.stderr)
