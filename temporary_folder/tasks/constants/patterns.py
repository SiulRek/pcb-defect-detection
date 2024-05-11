from enum import Enum
import re

from temporary_folder.tasks.constants.definitions import FILE_TAG
# ----------------- General  -------------------------

# ----------------- For Create Query -----------------

class CREATE_QUERY_PATTERNS(Enum):
    # For Line Validation for refererences
    FILE = re.compile(rf"#\s*((?:\S+\.(?:py|txt|log|md|csv))\s*(?:,\s*\S+\.(?:py|txt|log|md|csv)\s*)*|{FILE_TAG})")
    FILE_WITH_DIR = re.compile(r"#\s([\w/\\.-]+[\\/][\w.-]+\.(py|txt|log|md|csv))")
    FILL_TEXT = re.compile(r'^#\*\s*(.*)')
    RUN_SCRIPT = re.compile(rf"#run (\S+\.py|{FILE_TAG})")
    RUN_PYLINT = re.compile(rf"#pylint (\S+\.py|{FILE_TAG})")
    UNITTEST = re.compile(rf"#unittest (\S+\.py|{FILE_TAG})")
    DIRECTORY_TREE = re.compile(r"#tree (\S+)")
    SUMMARIZE_PYTHON_SCRIPT = re.compile(rf"#summarize (\S+\.py|{FILE_TAG})")
    SUMMARIZE_FOLDER = re.compile(r"#summarize_folder (\S+)")
    QUERY_TEMPLATE = re.compile(r'#(.*?)_query')
    CHECKSUM = re.compile(r"#checksum (\S+)")
    
    # For extracting error messages
    TEST_RESULT = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (ERROR|INFO) - .*)")