import re

from temporary_folder.tasks.constants.getters import get_python_environment_path
from temporary_folder.tasks.constants.patterns import (
    FILE_PATTERN,
    FILL_TEXT_PATTERN,
    RUN_SCRIPT_PATTERN,
    RUN_PYLINT_PATTERN,
)
from temporary_folder.tasks.constants.definitions import (
    TITLE_TAG,
    COMMENT_TAG,
    CURRENT_FILE_TAG,
    ERROR_TAG,
    REFERENCE_TYPE,
)
from temporary_folder.tasks.helpers.general.file_finder import file_finder
from temporary_folder.tasks.helpers.for_load_file_and_references.get_error_text import (
    get_error_text,
)
from temporary_folder.tasks.helpers.for_load_file_and_references.get_fill_text import (
    get_fill_text,
)
from temporary_folder.tasks.helpers.general.execute_python_script import (
    execute_python_script,
)
from temporary_folder.tasks.helpers.general.execute_pylint import execute_pylint


def handle_referenced_title(line):
    """Extract title from a line."""
    if TITLE_TAG in line:
        return (REFERENCE_TYPE.TITLE, line.replace(TITLE_TAG, "").strip())
    return None


def handle_referenced_comment(line):
    """Extract comments from a line."""
    if COMMENT_TAG in line:
        return (REFERENCE_TYPE.COMMENT, line.replace(COMMENT_TAG, "").strip())
    return None


def handle_referenced_files(line, root_dir, current_file_path):
    """Handle file pattern extraction and fetch file content."""
    match = re.search(FILE_PATTERN, line)
    if match:
        referenced_files = []
        file_names = match.group(1).split(",")
        file_names = [file_name.strip() for file_name in file_names]
        for file_name in file_names:
            file_path = file_finder(file_name, root_dir, current_file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                referenced_file = (REFERENCE_TYPE.FILE, (file_path, file.read()))
                referenced_files.append(referenced_file)
        return referenced_files
    return None


def handle_referenced_error(line, root_dir, current_file_path):
    """Extract error information based on tags."""
    if ERROR_TAG in line:
        error_text = get_error_text(root_dir, current_file_path)
        return (REFERENCE_TYPE.LOGGED_ERROR, error_text)
    return None


def handle_fill_text(line, root_dir):
    """Extract the fill text tag."""
    if match := FILL_TEXT_PATTERN.match(line):
        placeholder = match.group(1)
        fill_text, title = get_fill_text(placeholder, root_dir)
        return (REFERENCE_TYPE.FILL_TEXT, (fill_text, title))
    return None


def handle_run_python_script(line, root_dir, current_file_path):
    """Extract the run python script tag."""
    if match := RUN_SCRIPT_PATTERN.match(line):
        script_name = match.group(1)
        script_path = file_finder(script_name, root_dir, current_file_path)
        environment_path = get_python_environment_path(root_dir)
        script_output = execute_python_script(script_path, environment_path)
        return (REFERENCE_TYPE.RUN_PYTHON_SCRIPT, script_output)
    return None


def handle_run_pylint(line, root_dir, current_file_path):
    """Extract the run pylint tag."""
    if match := RUN_PYLINT_PATTERN.match(line):
        script_name = match.group(1)
        script_path = file_finder(script_name, root_dir, current_file_path)
        environment_path = get_python_environment_path(root_dir)
        pylint_output = execute_pylint(script_path, environment_path)
        return (REFERENCE_TYPE.RUN_PYLINT, pylint_output)
    return None


def handle_current_file_reference(line):
    """Extract the current file tag."""
    if CURRENT_FILE_TAG in line:
        return (REFERENCE_TYPE.CURRENT_FILE, None)
    return None


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
            if referenced_title := handle_referenced_title(stripped_line):
                referenced_contents.append(referenced_title)
            elif referenced_comment := handle_referenced_comment(stripped_line):
                if (
                    len(referenced_contents)
                    and referenced_contents[-1][0] == REFERENCE_TYPE.COMMENT
                ):
                    updated_comment = (
                        f"{referenced_contents[-1][1]}\n{referenced_comment[1]}"
                    )
                    referenced_contents[-1] = (REFERENCE_TYPE.COMMENT, updated_comment)
                else:
                    referenced_contents.append(referenced_comment)
            elif referenced_files := handle_referenced_files(
                stripped_line, root_dir, file_path
            ):
                referenced_contents.extend(referenced_files)
            elif referenced_error := handle_referenced_error(
                stripped_line, root_dir, file_path
            ):
                referenced_contents.append(referenced_error)
            elif referenced_fill_text := handle_fill_text(stripped_line, root_dir):
                referenced_contents.append(referenced_fill_text)
            elif referenced_run_script := handle_run_python_script(
                stripped_line, root_dir, file_path
            ):
                referenced_contents.append(referenced_run_script)
            elif referenced_run_pylint := handle_run_pylint(
                stripped_line, root_dir, file_path
            ):
                referenced_contents.append(referenced_run_pylint)
            elif current_file_tag := handle_current_file_reference(stripped_line):
                referenced_contents.append(current_file_tag)
            else:
                content_lines.append(line)

    non_referenced_content = "".join(content_lines)
    return referenced_contents, non_referenced_content
