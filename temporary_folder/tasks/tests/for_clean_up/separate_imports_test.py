import os

from temporary_folder.tasks.helpers.for_cleanup.separate_imports import separate_imports


TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEST_FILE_PATH = os.path.join(TEST_DIR, "data", "example_script_3.py")

def test_separate_imports():
    import_list, other_list = separate_imports(TEST_FILE_PATH)

    expected_imports = [
        'import os',
        'import sys',
        'from math import (\n    sqrt,\n    ceil,\n    floor\n)',
        'from datetime import datetime, timedelta',
        'import numpy as np'
    ]

    expected_others = [
        '# This is a comment',
        'x = 10',
        'print("Hello, world!")',
        'def function():\n    print("This is inside a function")',
        '# Another comment',
        'y = x + 5'
    ]


    assert import_list == expected_imports, f"Expected {expected_imports} but got {import_list}"
    assert other_list == expected_others, f"Expected {expected_others} but got {other_list}"

    print("All tests passed!")

test_separate_imports()