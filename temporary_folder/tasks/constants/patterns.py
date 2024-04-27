import re

# Reference File Pattern
FILE_PATTERN = re.compile(r"#\s*(\S+\.(py|txt|log|md|csv))")
FILE_PATTERN_WITH_DIR = re.compile(r"#\s([\w/\\.-]+[\\/][\w.-]+\.(py|txt|log|md|csv))")

# Test Result Pattern
TEST_RESULT_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (ERROR|INFO) - .*)")