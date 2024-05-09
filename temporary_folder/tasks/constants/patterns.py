import re

# Reference File Pattern
FILE_PATTERN = re.compile(r"#\s*((?:\S+\.(?:py|txt|log|md|csv))\s*(?:,\s*\S+\.(?:py|txt|log|md|csv)\s*)*)")

FILE_WITH_DIR_PATTERN = re.compile(r"#\s([\w/\\.-]+[\\/][\w.-]+\.(py|txt|log|md|csv))")

# Test Result Pattern
TEST_RESULT_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (ERROR|INFO) - .*)")

# Fill Text Pattern 
FILL_TEXT_PATTERN = re.compile(r'^#\*\s*(.*)')

# Run Python Script Name Pattern
RUN_SCRIPT_PATTERN = re.compile(r"#run (\S+\.py)")

# Run Pylint Pattern
RUN_PYLINT_PATTERN = re.compile(r"#pylint (\S+\.py)")

# Unittest Pattern
UNITTEST_PATTERN = re.compile(r"#unittest (\S+\.py)")

# Directory Tree Pattern
DIRECTORY_TREE_PATTERN = re.compile(r"#tree (\S+)")

# Summarize Python Script Pattern
SUMMARIZE_PYTHON_SCRIPT_PATTERN = re.compile(r"#summarize (\S+\.py)")

# Summarize Python Scripts in Folder Pattern
SUMMARIZE_FOLDER_PATTERN = re.compile(r"#summarize_folder (\S+)")

# Checksum Pattern
CHECKSUM_PATTERN = re.compile(r"#checksum (\S+)")