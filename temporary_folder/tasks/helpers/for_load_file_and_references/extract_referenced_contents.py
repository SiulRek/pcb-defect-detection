import re
import os

from temporary_folder.tasks.constants.getters import get_python_environment_path
from temporary_folder.tasks.constants.definitions import (
    REFERENCE_TYPE,
)
from temporary_folder.tasks.helpers.general.generate_directory_tree import (
    generate_directory_tree,
)
from temporary_folder.tasks.helpers.general.find_file import find_file
from temporary_folder.tasks.helpers.general.find_dir import find_dir
from temporary_folder.tasks.helpers.for_load_file_and_references.get_error_text import (
    get_error_text,
)
from temporary_folder.tasks.helpers.for_load_file_and_references.summarize_python_script import (
    summarize_python_file,
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
    line_validation_for_summarize_python_script,
)


def handle_referenced_title(line, *unused_args):
    """Extract title from a line."""
    if result := line_validation_for_title(line):
        return (REFERENCE_TYPE.TITLE, result, None)
    return None


def handle_referenced_comment(line, *unused_args):
    """Extract comments from a line."""
    if result := line_validation_for_comment(line):
        default_title = "Comment"
        return (REFERENCE_TYPE.COMMENT, default_title, result)
    return None


def handle_referenced_files(line, root_dir, current_file_path):
    """Handle file pattern extraction and fetch file content."""
    if result := line_validation_for_files(line):
        referenced_files = []
        for file_name in result:
            file_path = find_file(file_name, root_dir, current_file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                relative_path = os.path.relpath(file_path, root_dir)
                default_title = f"File at {relative_path}"
                referenced_file = (REFERENCE_TYPE.FILE, default_title, file.read())
                referenced_files.append(referenced_file)
        return referenced_files
    return None


def handle_referenced_error(line, root_dir, current_file_path):
    """Extract error information based on tags."""
    if line_validation_for_error(line):
        error_text = get_error_text(root_dir, current_file_path)
        default_title = "Occured Errors"
        return (REFERENCE_TYPE.LOGGED_ERROR, default_title, error_text)
    return None


def handle_fill_text(line, root_dir, *unused_args):
    """Extract the fill text tag."""
    if result := line_validation_for_fill_text(line):
        fill_text, default_title = get_fill_text(result, root_dir)
        return (REFERENCE_TYPE.FILL_TEXT, default_title, fill_text)


def handle_run_python_script(line, root_dir, current_file_path):
    """Extract the run python script tag."""
    if result := line_validation_for_run_python_script(line):
        script_path = find_file(result, root_dir, current_file_path)
        environment_path = get_python_environment_path(root_dir)
        script_output = execute_python_script(script_path, environment_path)
        default_title = "Python Script Output"
        return (REFERENCE_TYPE.RUN_PYTHON_SCRIPT, default_title, script_output)
    return None


def handle_run_pylint(line, root_dir, current_file_path):
    """Extract the run pylint tag."""
    if result := line_validation_for_run_pylint(line):
        script_path = find_file(result, root_dir, current_file_path)
        environment_path = get_python_environment_path(root_dir)
        pylint_output = execute_pylint(script_path, environment_path)
        default_title = "Pylint Output"
        return (REFERENCE_TYPE.RUN_PYLINT, default_title, pylint_output)
    return None


def handle_directory_tree(line, root_dir, current_file_path):
    """Extract the directory tree tag."""
    if result := line_validation_for_directory_tree(line):
        dir, max_depth, include_files, ignore_list = result
        dir = find_dir(dir, root_dir, current_file_path)
        directory_tree = generate_directory_tree(
            dir, max_depth, include_files, ignore_list
        )
        default_title = "Directory Tree"
        return (REFERENCE_TYPE.DIRECTORY_TREE, default_title, directory_tree)
    return None


def handle_summarize_python_script(line, root_dir, current_file_path):
    """Extract the summarize python script tag."""
    if result := line_validation_for_summarize_python_script(line):
        script_path = find_file(result, root_dir, current_file_path)
        script_summary = summarize_python_file(script_path)
        default_title = f"Summorized Python Script {os.path.basename(script_path)}"
        return (REFERENCE_TYPE.SUMMARIZE_PYTHON_SCRIPT, default_title, script_summary)
    return None


def handle_current_file_reference(line, root_dir, current_file_path):
    """Extract the current file tag."""
    if line_validation_for_current_file_reference(line):
        relative_path = os.path.relpath(current_file_path, root_dir)
        default_title = f"File at {relative_path}"
        return (REFERENCE_TYPE.CURRENT_FILE, default_title, None)
    return None


handlers = [
    handle_referenced_title,
    handle_referenced_comment,
    handle_referenced_files,
    handle_referenced_error,
    handle_fill_text,
    handle_run_python_script,
    handle_run_pylint,
    handle_directory_tree,
    handle_summarize_python_script,
    handle_current_file_reference,
]


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
            result = None
            stripped_line = line.strip()
            for handler in handlers:
                if result := handler(stripped_line, root_dir, file_path):
                    if isinstance(result, list):
                        referenced_contents.extend(result)
                    else:
                        referenced_contents.append(result)
                    break
            if not result:
                content_lines.append(line)

    for referenced_content in referenced_contents:
        if referenced_content[0] == REFERENCE_TYPE.COMMENT:
            start = referenced_contents.index(referenced_content)
            index = start + 1
            while (
                index < len(referenced_contents)
                and referenced_contents[index][0] == REFERENCE_TYPE.COMMENT
            ):
                merged_text = f"{referenced_content[2].strip()}\n"
                merged_text += f"{referenced_contents[index][2].strip()}"
                referenced_content = (
                    referenced_content[0],
                    referenced_content[1],
                    merged_text,
                )
                referenced_contents.pop(index)
            referenced_contents[start] = referenced_content

    non_referenced_content = "".join(content_lines)
    return referenced_contents, non_referenced_content
