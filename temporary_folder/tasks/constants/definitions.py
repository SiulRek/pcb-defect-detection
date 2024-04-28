# Process File Definitions

START_TAG = "#S "
END_TAG = "#E "
ERROR_TAG = "#L"    # As the Errors are derived from Logged Test Results 
TITLE_TAG = "#T "
COMMENT_TAG = "#C "
CURRENT_FILE_TAG = "#File"

# Enums
from enum import Enum

class REFERENCE_TYPE(Enum):
    COMMENT = "comment"
    FILE = "reference"
    LOGGED_ERROR = "error"
    CURRENT_FILE = "current_file"
    RUN_PYTHON_SCRIPT = "run_python_script"
    RUN_PYLINT = "run_pylint"
    FILL_TEXT = "fill_text"
    TITLE = "title"
    
# General Constants
TEST_RESULTS_FILE = "test_results.log"