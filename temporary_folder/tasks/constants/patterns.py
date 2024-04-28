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