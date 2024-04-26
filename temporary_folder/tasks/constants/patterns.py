import re

# Reference File Pattern
FILE_PATTERN = re.compile(r"#\s*(\S+\.(py|txt|log|md|csv))")
FILE_PATTERN_WITH_DIR = re.compile(r"#\s([\w/\\.-]+[\\/][\w.-]+\.(py|txt|log|md|csv))")