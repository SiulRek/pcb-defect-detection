import re

from temporary_folder.tasks.constants.patterns import (
    FILE_PATTERN,
    FILL_TEXT_PATTERN,
    RUN_SCRIPT_PATTERN,
    RUN_PYLINT_PATTERN,
    UNITTEST_PATTERN,
    DIRECTORY_TREE_PATTERN,
    SUMMARIZE_PYTHON_SCRIPT_PATTERN,
    SUMMARIZE_FOLDER_PATTERN,
    CHECKSUM_PATTERN,
)
from temporary_folder.tasks.constants.definitions import (
    START_TAG,
    END_TAG,
    TITLE_TAG,
    COMMENT_TAG,
    CURRENT_FILE_TAG,
    ERROR_TAG,
    MAKE_QUERY_TAG,
)
from temporary_folder.tasks.constants.defaults import DIRECTORY_TREE_DEFAULTS
from temporary_folder.tasks.helpers.general.line_validation_utils import (
    retrieve_list_in_square_brackets,
    retrieve_optional_arguments,
    retrieve_bool,
)


def line_validation_for_start_tag(line):
    """ Validate if the line is a start tag."""
    if START_TAG in line:
        return line.split(START_TAG, 1)[1].strip()
    return None

    
def line_validation_for_end_tag(line):
    """ Validate if the line is an end tag."""
    if END_TAG in line:
        return line.split(END_TAG, 1)[1].strip()
    return None


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


def line_validation_for_current_file_reference(line):
    """ Validate if the line is a current file reference."""
    if CURRENT_FILE_TAG in line:
        return True
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


def line_validation_for_run_unittest(line):
    """ Validate if the line is a run unittest."""
    if match := re.search(UNITTEST_PATTERN, line):
        verbosity = 1
        if arguments := retrieve_optional_arguments(line):
            verbosity = int(arguments[0])
        return match.group(1), verbosity
    return None


def line_validation_for_directory_tree(line):
    """ Validate if the line is a directory tree."""
    if match := DIRECTORY_TREE_PATTERN.match(line):
        dir = match.group(1)
        max_depth = DIRECTORY_TREE_DEFAULTS.MAX_DEPTH.value
        include_files = DIRECTORY_TREE_DEFAULTS.INCLUDE_FILES.value
        ignore_list = DIRECTORY_TREE_DEFAULTS.IGNORE_LIST.value
        if arguments := retrieve_optional_arguments(line):
            if len(arguments) >= 1:
                max_depth = int(arguments[0])
            if len(arguments) >= 2:
                include_files = retrieve_bool(arguments[1])
            if len(arguments) == 3:
                err_msg = "Invalid ignore list in directory tree"
                ignore_list = retrieve_list_in_square_brackets(arguments[2], err_msg)
        return (dir, max_depth, include_files, ignore_list)
    return None


def line_validation_for_summarize_python_script(line):
    """ Validate if the line is a summarize python script."""
    if match := SUMMARIZE_PYTHON_SCRIPT_PATTERN.match(line):
        include_definitions_without_docstrings = False
        if arguments := retrieve_optional_arguments(line):
            bool = True if arguments[0].lower() == "true" else False
            include_definitions_without_docstrings = bool
        return match.group(1), include_definitions_without_docstrings
    return None


def line_validation_for_summarize_folder(line):
    """ Validate if the line is a summarize folder."""
    if match := SUMMARIZE_FOLDER_PATTERN.match(line):
        dir = match.group(1)
        include_definitions_without_docstrings = False
        err_msg = "Invalid summarize folder arguments"
        excluded_dirs = []
        excluded_files = []
        if arguments := retrieve_optional_arguments(line):
            include_definitions_without_docstrings = retrieve_bool(arguments[0])
            if len(arguments) > 1:
                excluded_dirs = retrieve_list_in_square_brackets(arguments[1], err_msg)
            if len(arguments) > 2:
                excluded_files = retrieve_list_in_square_brackets(arguments[2], err_msg)
        return dir, include_definitions_without_docstrings, excluded_dirs, excluded_files
    

def line_validation_for_make_query(line):
    """
    Validate the line to check if it is a valid line to make a query.
    """
    if MAKE_QUERY_TAG in line:
        max_tokens = None
        create_python_script = False
        if arguments := retrieve_optional_arguments(line):
            create_python_script = retrieve_bool(arguments[0])
            if len(arguments) > 1:
                max_tokens = int(arguments[1])
        return True, create_python_script, max_tokens
    return None


def line_validation_for_checksum(line):
    """
    Validate the line to check if it is a valid line to add checksum.
    """
    if result := CHECKSUM_PATTERN.match(line):
        try:
            checksum = int(result.group(1).strip())
            return checksum
        except ValueError as e:
            msg = "Invalid checksum line. Must be in the format: #checksum (checksum)"
            raise ValueError(msg) from e
    return None