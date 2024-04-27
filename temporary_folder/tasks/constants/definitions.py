# Process File Definitions

START_TAG = "#S "
END_TAG = "#E "
ERROR_TAG = "#Er"
COMMENT_TAG = "#C "

# Enums
from enum import Enum

class REFERENCE_TYPE(Enum):
    COMMENT = "comment"
    FILE = "reference"
    OTHER = "other"

# General Constants
TEST_RESULTS_FILE = "test_results.log"