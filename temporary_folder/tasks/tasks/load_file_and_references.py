import sys
import os
from copy import deepcopy

ROOT_DIR = sys.argv[1]
FILE_PATH = sys.argv[2]
sys.path.append(ROOT_DIR)
from temporary_folder.tasks.helpers.for_load_file_and_references.extract_start_and_end_text import (
    extract_start_and_end_text,
)
from temporary_folder.tasks.helpers.for_load_file_and_references.extract_referenced_contents import (
    extract_referenced_contents,
)
from temporary_folder.tasks.helpers.for_load_file_and_references.add_text_tags import (
    add_text_tags,
)
from temporary_folder.tasks.constants.getters import get_temporary_file_path
from temporary_folder.tasks.constants.definitions import REFERENCE_TYPE
import temporary_folder.tasks.helpers.general.print_statements as task_prints


TEMPORARY_FILE = get_temporary_file_path(ROOT_DIR)


class ReferenceTitleManager:

    def __init__(self):
        self.title = None

    def set(self, title):
        self.title = title

    def get(self):
        title = deepcopy(self.title)
        self.title = None
        return title


def format_text_from_references(
    referenced_contents, file_path, updated_content, root_dir
):
    """
    Formats a query string from file references and updated content.

    Args:
        referenced_contents (list): A list of tuples detailing references (type, data).
        file_path (str): The path to the current file.
        updated_content (str): The updated content of the current file.
        root_dir (str): The root directory of the project.

    Returns:
        str: Formatted query based on file references.
    """
    if not any(
        content_type == REFERENCE_TYPE.CURRENT_FILE
        for content_type, _ in referenced_contents
    ):
        referenced_contents.insert(0, (REFERENCE_TYPE.CURRENT_FILE, None))

    query = ""
    relative_path = os.path.relpath(file_path, root_dir)
    query += f"--- File at: {relative_path} ---\n{updated_content}"
    current_file_path = file_path
    title_manager = ReferenceTitleManager()
    for content_type, data in referenced_contents:
        current_title = title_manager.get()

        if content_type == REFERENCE_TYPE.TITLE:
            title_manager.set(data)
        elif content_type == REFERENCE_TYPE.COMMENT:
            title = current_title or "Comment"
            query += f"\n\n--- {title} ---\n{data}"
        elif content_type == REFERENCE_TYPE.FILE:
            file_path, file_content = data
            relative_path = os.path.relpath(file_path, root_dir)
            title = current_title or f"File at: {relative_path}"
            query += f"\n\n--- {title} ---\n{file_content}"
        elif content_type == REFERENCE_TYPE.LOGGED_ERROR:
            title = current_title or "Occurred Error"
            query += f"\n\n--- {title} ---\n{data}"
        elif content_type == REFERENCE_TYPE.FILL_TEXT:
            fill_text, text_title = data
            title = current_title or text_title
            query += f"\n\n--- {title} ---\n{fill_text}"
        elif content_type == REFERENCE_TYPE.CURRENT_FILE:
            relative_path = os.path.relpath(current_file_path, root_dir)
            title = current_title or f"File at: {relative_path}"
            query += f"\n\n--- {title} ---\n{updated_content}"
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    return query


def load_file_and_references(file_path, root_dir, query_path):
    """
    Load the content of the specified Python file and additionally load any files
    referenced on lines with a '#' symbol.

    Args:
        file_path (str): The path to the Python file.
        root_dir (str): The root directory of the project.
        query_path (str): The path to the file where the query will be saved.
    """
    referenced_contents, updated_content = extract_referenced_contents(
        file_path, root_dir
    )
    start_text, updated_content, end_text = extract_start_and_end_text(
        file_path, updated_content
    )

    query = format_text_from_references(
        referenced_contents, file_path, updated_content, root_dir
    )

    query = add_text_tags(start_text, end_text, query)

    with open(query_path, "w", encoding="utf-8") as output_file:
        output_file.write(query)

    print(f"Query generated and saved to: {query_path}")


def main():
    task_prints.process_start("Load File and References")
    load_file_and_references(FILE_PATH, ROOT_DIR, TEMPORARY_FILE)
    task_prints.process_end()


if __name__ == "__main__":
    main()