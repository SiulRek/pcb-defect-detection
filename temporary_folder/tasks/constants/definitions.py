from enum import Enum


# ----------------- General Constants -----------------
TEST_RESULTS_FILE = "test_results.log"
FILE_TAG = "File"


# ----------------- For Make Query -----------------
START_TAG = "#S "
END_TAG = "#E "
ERROR_TAG = "#L"    # As the Errors are derived from Logged Test Results 
TITLE_TAG = "#T "
COMMENT_TAG = "#C "
CURRENT_FILE_TAG = "#File"
MAKE_QUERY_TAG = "#makequery"

class MAKE_QUERY_REFERENCE_TYPES(Enum):
    COMMENT = "comment"
    FILE = "reference"
    LOGGED_ERROR = "error"
    CURRENT_FILE = "current_file"
    RUN_PYTHON_SCRIPT = "run_python_script"
    RUN_PYLINT = "run_pylint"
    RUN_UNITTEST = "run_unittest"
    DIRECTORY_TREE = "directory_tree"
    SUMMARIZE_PYTHON_SCRIPT = "summarize_python_script"
    SUMMARIZE_FOLDER = "summarize_folder" # Summarize all Python scripts in a folder
    FILL_TEXT = "fill_text"
    TITLE = "title"

# ----------------- For Clean Up -----------------
LINE_WIDTH = 80
INTEND = " " * 4
DOC_QUOTE = '"""'