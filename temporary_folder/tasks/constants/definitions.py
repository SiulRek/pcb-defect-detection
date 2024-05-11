from enum import Enum


# ----------------- General Constants -----------------
TEST_RESULTS_FILE = "test_results.log"
FILE_TAG = "File"


# ----------------- For Create Query -----------------
MAKE_QUERY_TAG = "#makequery"

class CREATE_QUERY_TAGS(Enum):
    BEGIN = "#B "
    END = "#E "
    ERROR = "#L"
    TITLE = "#T "
    COMMENT = "#C "
    CURRENT_FILE = f"#{FILE_TAG}"
    MAKE_QUERY = "#makequery"


class MAKE_QUERY_REFERENCE_TYPES(Enum):
    BEGIN_TEXT = "begin_text",
    END_TEXT = "end_text",
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
    MAKE_QUERY = "make_query"

# ----------------- For Clean Up -----------------
LINE_WIDTH = 80
INTEND = " " * 4
DOC_QUOTE = '"""'

class CLEANUP_TAGS(Enum):
    SELECT_ONLY = "#only"
    SELECT_NOT = "#not"
    CHECKPOINTS = "#checkpointing"

class CLEANUP_REFERENCE_TYPES(Enum):
    SELECT_ONLY = "select_only"
    SELECT_NOT = "select_not"
    CHECKPOINTING = "checkpointing"