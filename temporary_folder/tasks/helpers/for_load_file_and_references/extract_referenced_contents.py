import re

from temporary_folder.tasks.constants.getters import get_python_environment_path
from temporary_folder.tasks.constants.definitions import (
    REFERENCE_TYPE,
)
from temporary_folder.tasks.helpers.general.generate_directory_tree import (
    generate_directory_tree
)
from temporary_folder.tasks.helpers.general.find_file import find_file
from temporary_folder.tasks.helpers.general.find_dir import find_dir
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

from temporary_folder.tasks.helpers.for_load_file_and_references.line_validation import (
    line_validation_for_title,
    line_validation_for_comment,
    line_validation_for_files,
    line_validation_for_error,
    line_validation_for_fill_text,
    line_validation_for_run_python_script,
    line_validation_for_run_pylint,
    line_validation_for_current_file_reference,
    line_validation_for_directory_tree,
)


def handle_referenced_title(line):
    """Extract title from a line."""
    if result := line_validation_for_title(line):
        return (REFERENCE_TYPE.TITLE, result)
    return None


def handle_referenced_comment(line):
    """Extract comments from a line."""
    if result := line_validation_for_comment(line):
        return (REFERENCE_TYPE.COMMENT, result)
    return None


def handle_referenced_files(line, root_dir, current_file_path):
    """Handle file pattern extraction and fetch file content."""
    if result := line_validation_for_files(line):
        referenced_files = []
        for file_name in result:
            file_path = find_file(file_name, root_dir, current_file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                referenced_file = (REFERENCE_TYPE.FILE, (file_path, file.read()))
                referenced_files.append(referenced_file)
        return referenced_files
    return None


def handle_referenced_error(line, root_dir, current_file_path):
    """Extract error information based on tags."""
    if line_validation_for_error(line):
        error_text = get_error_text(root_dir, current_file_path)
        return (REFERENCE_TYPE.LOGGED_ERROR, error_text)
    return None


def handle_fill_text(line, root_dir):
    """Extract the fill text tag."""
    if result := line_validation_for_fill_text(line):
        fill_text = get_fill_text(result, root_dir)
        return (REFERENCE_TYPE.FILL_TEXT, fill_text)


def handle_run_python_script(line, root_dir, current_file_path):
    """Extract the run python script tag."""
    if result := line_validation_for_run_python_script(line):
        script_path = find_file(result, root_dir, current_file_path)
        environment_path = get_python_environment_path(root_dir)
        script_output = execute_python_script(script_path, environment_path)
        return (REFERENCE_TYPE.RUN_PYTHON_SCRIPT, script_output)
    return None


def handle_run_pylint(line, root_dir, current_file_path):
    """Extract the run pylint tag."""
    if result := line_validation_for_run_pylint(line):
        script_path = find_file(result, root_dir, current_file_path)
        environment_path = get_python_environment_path(root_dir)
        pylint_output = execute_pylint(script_path, environment_path)
        return (REFERENCE_TYPE.RUN_PYLINT, pylint_output)
    return None


def handle_directory_tree(line, root_dir, current_file_path):
    """Extract the directory tree tag."""
    if result := line_validation_for_directory_tree(line):
        dir, max_depth, include_files, ignore_list = result
        dir = find_dir(dir, root_dir, current_file_path)
        directory_tree = generate_directory_tree(
            dir, max_depth, include_files, ignore_list
        )
        return (REFERENCE_TYPE.DIRECTORY_TREE, directory_tree)
    return None


def handle_current_file_reference(line):
    """Extract the current file tag."""
    if line_validation_for_current_file_reference(line):
        return (REFERENCE_TYPE.CURRENT_FILE, line)
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
            elif referenced_directory_tree := handle_directory_tree(
                stripped_line, root_dir, file_path
            ):
                referenced_contents.append(referenced_directory_tree)
            elif current_file_tag := handle_current_file_reference(stripped_line):
                referenced_contents.append(current_file_tag)
            else:
                content_lines.append(line)

    non_referenced_content = "".join(content_lines)
    return referenced_contents, non_referenced_content
