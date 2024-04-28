import re

from temporary_folder.tasks.constants.patterns import (
    FILE_PATTERN,
    FILL_TEXT_PATTERN,
)
from temporary_folder.tasks.constants.definitions import (
    COMMENT_TAG,
    CURRENT_FILE_TAG,
    ERROR_TAG,
    REFERENCE_TYPE,
)
from temporary_folder.tasks.helpers.general.file_finder import file_finder
from temporary_folder.tasks.helpers.for_load_file_and_references.get_error_text import get_error_text
from temporary_folder.tasks.helpers.for_load_file_and_references.get_fill_text import get_fill_text


def handle_referenced_comment(line):
    """Extract comments from a line."""
    if COMMENT_TAG in line:
        return (REFERENCE_TYPE.COMMENT, line.replace(COMMENT_TAG, "").strip())


def handle_referenced_file(line, root_dir, current_file_path):
    """Handle file pattern extraction and fetch file content."""
    match = re.search(FILE_PATTERN, line)
    if match:
        referenced_file_path = file_finder(match.group(1), root_dir, current_file_path)
        with open(referenced_file_path, "r", encoding="utf-8") as ref_file:
            file_contents = ref_file.read()
            return (REFERENCE_TYPE.FILE, (referenced_file_path, file_contents))
        

def handle_referenced_error(line, root_dir, current_file_path):
    """Extract error information based on tags."""
    if ERROR_TAG in line:
        error_text = get_error_text(root_dir, current_file_path)
        return (REFERENCE_TYPE.LOGGED_ERROR, error_text)

def handle_fill_text(line, root_dir):
    """Extract the fill text tag."""
    if match := FILL_TEXT_PATTERN.match(line):
        placeholder = match.group(1)
        fill_text, title = get_fill_text(placeholder, root_dir)
        return (REFERENCE_TYPE.FILL_TEXT, (fill_text, title))
    

def handle_current_file_reference(line):
    """Extract the current file tag."""
    if CURRENT_FILE_TAG in line:
       return (REFERENCE_TYPE.CURRENT_FILE, None)


def extract_referenced_contents(file_path, root_dir):
    """
    Extracts referenced files, comments, and other content from a specified file, maintaining
    the order of their occurrence.

    Args:
        file_path (str): The path to the file.
        root_dir (str): The root directory of the project.
    
    Returns:
        tuple: A tuple containing a list of referenced contents and the non-referenced content.
    """
    referenced_contents = []
    content_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if referenced_comment := handle_referenced_comment(stripped_line):
                referenced_contents.append(referenced_comment)
            elif referenced_file := handle_referenced_file(stripped_line, root_dir, file_path):
                referenced_contents.append(referenced_file)
            elif referenced_error := handle_referenced_error(stripped_line, root_dir, file_path):
                referenced_contents.append(referenced_error)
            elif referenced_fill_text :=  handle_fill_text(stripped_line, root_dir):
                referenced_contents.append(referenced_fill_text)
            elif current_file_tag := handle_current_file_reference(stripped_line):
                referenced_contents.append(current_file_tag)
            else:
                content_lines.append(line)

    non_referenced_content = "".join(content_lines)
    return referenced_contents, non_referenced_content

