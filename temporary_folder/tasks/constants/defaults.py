from enum import Enum


class DIRECTORY_TREE_DEFAULTS(Enum):
    """Default values for the directory tree tag."""

    MAX_DEPTH = float("inf")  # Allows infinite depth unless specified otherwise.
    INCLUDE_FILES = False  # Does not include files in the directory tree by default.
    IGNORE_LIST = [
        "pcb-defect-detection.code-workspace",
        ".vscode",
        "venv",
        "__pycache__",
        "batch_scripts",
        "local",
        "temp.py",
        "test.py",
        "test.txt",
        "temp.txt",
        "temporary_folder" "test_results.log",
        "test_results_simple.log",
        "source/preprocessing/notebooks/related/pipeline.json",
        "references/private",
    ]
