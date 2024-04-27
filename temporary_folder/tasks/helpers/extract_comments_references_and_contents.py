import re

from temporary_folder.tasks.helpers.find_nearest_file import find_nearest_file
from temporary_folder.tasks.constants.patterns import (
    FILE_PATTERN,
    FILE_PATTERN_WITH_DIR,
)
from temporary_folder.tasks.constants.definitions import (
    COMMENT_TAG,
    REFERENCE_TYPE,
    ERROR_TAG,
)
from temporary_folder.tasks.helpers.find_file_from_path_fragment import (
    find_file_from_path_fragment,
)
from temporary_folder.tasks.helpers.get_error_text import get_error_text


def extract_content_references_and_comments(file_path, root_dir):
    """
    Extracts referenced files, comments, and other content from a specified file, maintaining
    the order of their occurrence.

    Args:
        file_path (str): The path to the Python file from which references and comments are extracted.
        root_dir (str): The root directory used to find the nearest referenced files.

    Returns:
        (list, str): A tuple containing the non-referenced content of the file and a list of tuples
        containing the type of reference (comment or file), the path to the referenced file (if applicable),
        and the content of the referenced file.
    """
    referenced_contents = []
    content_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if COMMENT_TAG in line:
                referenced_contents.append(
                    (REFERENCE_TYPE.COMMENT, line.replace(COMMENT_TAG, "").strip())
                )
            elif match := re.search(FILE_PATTERN, line):
                match_with_dir = re.search(FILE_PATTERN_WITH_DIR, line)
                if match_with_dir:
                    path_fragment = match_with_dir.group(1)
                    referenced_file_path = find_file_from_path_fragment(
                        path_fragment, root_dir
                    )
                else:
                    referenced_file_name = match.group(1)
                    referenced_file_path = find_nearest_file(
                        referenced_file_name, root_dir, file_path
                    )
                if referenced_file_path:
                    with open(referenced_file_path, "r", encoding="utf-8") as ref_file:
                        file_contents = ref_file.read()
                        referenced_contents.append(
                            (REFERENCE_TYPE.FILE, (referenced_file_path, file_contents))
                        )
            elif ERROR_TAG in line:
                error_text = get_error_text(root_dir, file_path)
                referenced_contents.append((REFERENCE_TYPE.LOGGED_ERROR, error_text))
            else:
                content_lines.append(line)

    non_referenced_content = "".join(content_lines)
    return non_referenced_content, referenced_contents
