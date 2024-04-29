import re

from temporary_folder.tasks.constants.patterns import (
    FILE_PATTERN,
    FILL_TEXT_PATTERN,
    RUN_SCRIPT_PATTERN,
    RUN_PYLINT_PATTERN,
    DIRECTORY_TREE_PATTERN,
    SUMMARIZE_PYTHON_SCRIPT_PATTERN,
)
from temporary_folder.tasks.constants.definitions import (
    TITLE_TAG,
    COMMENT_TAG,
    CURRENT_FILE_TAG,
    ERROR_TAG,
)
from temporary_folder.tasks.constants.defaults import DIRECTORY_TREE_DEFAULTS

ROUND_BRACKET_PATTERN = re.compile(r"\((.*?)\)")
SQUARE_BRACKET_PATTERN = re.compile(r"\[(.*?)\]")


def line_validation_for_title(line):
    """ Validate if the line is a title."""
    if TITLE_TAG in line:
        return line.replace(TITLE_TAG, "").strip()
    return None


def line_validation_for_comment(line):
    """ Validate if the line is a comment."""
    if COMMENT_TAG in line:
        return line.replace(COMMENT_TAG, "").strip()
    return None


def line_validation_for_files(line):
    """ Validate if the line is a file."""
    if match := re.search(FILE_PATTERN, line):
        file_names = match.group(1).split(",")
        file_names = [file_name.strip() for file_name in file_names]
        return file_names
    return None


def line_validation_for_error(line):
    """ Validate if the line is an error."""
    if ERROR_TAG in line:
        return True
    return None


def line_validation_for_fill_text(line):
    """ Validate if the line is a fill text."""
    if match := FILL_TEXT_PATTERN.match(line):
        placeholder = match.group(1)
        return placeholder
    return None


def line_validation_for_run_python_script(line):
    """ Validate if the line is a run python script."""
    if match := RUN_SCRIPT_PATTERN.match(line):
        return match.group(1)
    return None


def line_validation_for_run_pylint(line):
    """ Validate if the line is a run pylint."""
    if match := RUN_PYLINT_PATTERN.match(line):
        return match.group(1)
    return None


def line_validation_for_current_file_reference(line):
    """ Validate if the line is a current file reference."""
    if CURRENT_FILE_TAG in line:
        return True
    return None


def line_validation_for_directory_tree(line):
    """ Validate if the line is a directory tree."""
    if match := DIRECTORY_TREE_PATTERN.match(line):
        dir = match.group(1)
        max_depth = DIRECTORY_TREE_DEFAULTS.MAX_DEPTH.value
        include_files = DIRECTORY_TREE_DEFAULTS.INCLUDE_FILES.value
        ignore_list = DIRECTORY_TREE_DEFAULTS.IGNORE_LIST.value
        result = re.search(ROUND_BRACKET_PATTERN, line)
        if result:
            arguments = result.group(1).split(",")
            arguments = [arg.strip() for arg in arguments]
            if len(arguments) >= 1:
                max_depth = int(arguments[0])
            if len(arguments) >= 2:
                include_files = True if arguments[1].lower() == "true" else False
            if len(arguments) == 3:
                match = SQUARE_BRACKET_PATTERN.match(arguments[2])
                if not match:
                    raise ValueError("Invalid directory tree arguments")
                additional_ignore_list = match.group(1).split(";")
                ignore_list.extend([ignore.strip() for ignore in additional_ignore_list])
        return (dir, max_depth, include_files, ignore_list)
    return None

def line_validation_for_summarize_python_script(line):
    """ Validate if the line is a summarize python script."""
    if match := SUMMARIZE_PYTHON_SCRIPT_PATTERN.match(line):
        return match.group(1)
    return None