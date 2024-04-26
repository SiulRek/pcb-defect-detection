import re

from temporary_folder.tasks.helpers.find_nearest_file import find_nearest_file
from temporary_folder.tasks.constants.patterns import FILE_PATTERN


def extract_references_and_content(file_path, root_dir):
    """
    Extracts referenced Python files and content from a specified file.

    Args:
        file_path (str): The path to the Python file from which references are extracted.
        root_dir (str): The root directory used to find the nearest referenced files.

    Returns:
        tuple: A dictionary of referenced files and their contents, and a list of non-referenced content lines.
    """
    referenced_files = {}
    content_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "#" in line:
                match = re.search(FILE_PATTERN, line)
                if match:
                    referenced_file_name = match.group(1)
                    referenced_file_path = find_nearest_file(
                        referenced_file_name, root_dir, file_path
                    )
                    if referenced_file_path:
                        with open(
                            referenced_file_path, "r", encoding="utf-8"
                        ) as ref_file:
                            referenced_files[referenced_file_path] = ref_file.read()
                else:
                    content_lines.append(line)
            else:
                content_lines.append(line)

    contents = "".join(content_lines)
    return referenced_files, contents
