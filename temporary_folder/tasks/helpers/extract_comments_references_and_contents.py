import re

from temporary_folder.tasks.constants.patterns import (
    FILE_PATTERN,
    FILE_PATTERN_WITH_DIR,
)
from temporary_folder.tasks.constants.definitions import (
    COMMENT_TAG,
    CURRENT_FILE_TAG,
    ERROR_TAG,
    REFERENCE_TYPE,
)
from temporary_folder.tasks.helpers.file_finder import file_finder
from temporary_folder.tasks.helpers.get_error_text import get_error_text


def extract_comments(line):
    """Extract comments from a line."""
    if COMMENT_TAG in line:
        return (REFERENCE_TYPE.COMMENT, line.replace(COMMENT_TAG, "").strip())


def handle_file_pattern(line, root_dir, current_file_path):
    """Handle file pattern extraction and fetch file content."""
    match = re.search(FILE_PATTERN, line)
    if match:
        referenced_file_path = file_finder(match.group(1), root_dir, current_file_path)
        with open(referenced_file_path, "r", encoding="utf-8") as ref_file:
            file_contents = ref_file.read()
            return (REFERENCE_TYPE.FILE, (referenced_file_path, file_contents))
        

def handle_error_tags(line, root_dir, current_file_path):
    """Extract error information based on tags."""
    if ERROR_TAG in line:
        error_text = get_error_text(root_dir, current_file_path)
        return (REFERENCE_TYPE.LOGGED_ERROR, error_text)
    

def handle_current_file_tag(line):
    """Extract the current file tag."""
    if CURRENT_FILE_TAG in line:
       return (REFERENCE_TYPE.CURRENT_FILE, None)


def extract_content_references_and_comments(file_path, root_dir):
    """
    Extracts referenced files, comments, and other content from a specified file, maintaining
    the order of their occurrence.
    """
    referenced_contents = []
    content_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if referenced_comment := extract_comments(line):
                referenced_contents.append(referenced_comment)
            elif referenced_file := handle_file_pattern(line, root_dir, file_path):
                referenced_contents.append(referenced_file)
            elif referenced_error := handle_error_tags(line, root_dir, file_path):
                referenced_contents.append(referenced_error)
            elif current_file_tag := handle_current_file_tag(line):
                referenced_contents.append(current_file_tag)
            else:
                content_lines.append(line)

    non_referenced_content = "".join(content_lines)
    return non_referenced_content, referenced_contents

