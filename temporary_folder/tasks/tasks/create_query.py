import sys
import os

if len(sys.argv) == 3:
    ROOT_DIR = sys.argv[1]
    FILE_PATH = sys.argv[2]
    sys.path.append(ROOT_DIR)
else:
    ROOT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
    )
    FILE_PATH = os.path.join(
        ROOT_DIR,
        "temporary_folder",
        "tasks",
        "tests",
        "load_file_and_references_test.py",
    )


from temporary_folder.tasks.helpers.for_create_query.extract_start_and_end_text import (
    extract_start_and_end_text,
)
from temporary_folder.tasks.helpers.for_create_query.extract_referenced_contents import (
    extract_referenced_contents,
)
from temporary_folder.tasks.helpers.for_create_query.add_text_tags import (
    add_text_tags,
)
from temporary_folder.tasks.constants.getters import (
    get_temporary_file_path,
    get_response_file_path,
)
from temporary_folder.tasks.constants.definitions import REFERENCE_TYPE
import temporary_folder.tasks.helpers.general.print_statements as task_prints
from temporary_folder.tasks.helpers.for_create_query.finalizer import (
    Finalizer,
)

TEMPORARY_FILE = get_temporary_file_path(ROOT_DIR)
RESPONSE_FILE = get_response_file_path(ROOT_DIR)


class ReferenceTitleManager:

    def __init__(self):
        self.title = None

    def set(self, title):
        self.title = title

    def get(self):
        title, self.title = self.title, None
        return title


def format_text_from_references(referenced_contents, updated_content):
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
    query = ""
    title_manager = ReferenceTitleManager()
    for referenced_content in referenced_contents:
        content_type, default_title, text = referenced_content
        current_title = title_manager.get()
        title = current_title if current_title else default_title

        if content_type == REFERENCE_TYPE.TITLE:
            title_manager.set(default_title)
        elif content_type == REFERENCE_TYPE.CURRENT_FILE:
            query += f"\n\n--- {title} ---\n{updated_content}"
        elif content_type in REFERENCE_TYPE:
            query += f"\n\n--- {title} ---\n{text}"
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    return query


def create_query(file_path, root_dir, query_path, response_path):
    """
    Create a query from the file and referenced contents in the file.

    Args:
        file_path (str): The path to the file to be processed.
        root_dir (str): The root directory of the project.
        query_path (str): The path to the query file.
        response_path (str): The path to the response file.
    """
    referenced_contents, updated_content = extract_referenced_contents(
        file_path, root_dir
    )
    start_text, updated_content, end_text = extract_start_and_end_text(
        file_path, updated_content
    )

    query = format_text_from_references(referenced_contents, file_path)

    query = add_text_tags(start_text, end_text, query)

    finalizer = Finalizer()
    finalizer.set_paths(file_path, query_path, response_path)
    finalizer.finalize(updated_content, query)


def main():
    task_prints.process_start("Create Query")
    create_query(FILE_PATH, ROOT_DIR, TEMPORARY_FILE, RESPONSE_FILE)
    task_prints.process_end()


if __name__ == "__main__":
    main()