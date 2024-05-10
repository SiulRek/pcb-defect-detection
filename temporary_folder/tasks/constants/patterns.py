import re

from temporary_folder.tasks.constants.definitions import FILE_TAG

# Reference File Pattern
FILE_PATTERN = re.compile(rf"#\s*((?:\S+\.(?:py|txt|log|md|csv))\s*(?:,\s*\S+\.(?:py|txt|log|md|csv)\s*)*|{FILE_TAG})")

FILE_WITH_DIR_PATTERN = re.compile(r"#\s([\w/\\.-]+[\\/][\w.-]+\.(py|txt|log|md|csv))")

# Test Result Pattern
TEST_RESULT_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (ERROR|INFO) - .*)")

# Fill Text Pattern 
FILL_TEXT_PATTERN = re.compile(r'^#\*\s*(.*)')

# Run Python Script Name Pattern
RUN_SCRIPT_PATTERN = re.compile(rf"#run (\S+\.py|{FILE_TAG})")

# Run Pylint Pattern
RUN_PYLINT_PATTERN = re.compile(rf"#pylint (\S+\.py|{FILE_TAG})")

# Unittest Pattern
UNITTEST_PATTERN = re.compile(rf"#unittest (\S+\.py|{FILE_TAG})")

# Directory Tree Pattern
DIRECTORY_TREE_PATTERN = re.compile(r"#tree (\S+)")

# Summarize Python Script Pattern
SUMMARIZE_PYTHON_SCRIPT_PATTERN = re.compile(rf"#summarize (\S+\.py|{FILE_TAG})")

# Summarize Python Scripts in Folder Pattern
SUMMARIZE_FOLDER_PATTERN = re.compile(r"#summarize_folder (\S+)")

# Template Pattern
QUERY_TEMPLATE_PATTERN = re.compile(r'#(.*?)_query')

# Checksum Pattern
CHECKSUM_PATTERN = re.compile(r"#checksum (\S+)")
