import sys
import os

ROOT_DIR = sys.argv[1]
FILE_PATH = sys.argv[2]
sys.path.append(ROOT_DIR)
from temporary_folder.tasks.helpers.process_file import process_file
from temporary_folder.tasks.helpers.extract_comments_references_and_contents import (
    extract_content_references_and_comments,
)
from temporary_folder.tasks.helpers.add_text_tags import add_text_tags
from temporary_folder.tasks.constants.getters import get_temporary_file_path
from temporary_folder.tasks.constants.definitions import REFERENCE_TYPE
import temporary_folder.tasks.helpers.print_statements as task_prints


TEMPORARY_FILE = get_temporary_file_path(ROOT_DIR)


def format_text_from_references(referenced_contents, file_path, updated_content, root_dir):
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

    for content_type, data in referenced_contents:
        if content_type == REFERENCE_TYPE.COMMENT:
            query += f"\n\n--- Comment ---\n{data}"
        elif content_type == REFERENCE_TYPE.FILE:
            file_path, file_content = data
            relative_path = os.path.relpath(file_path, root_dir)
            query += f"\n\n--- File at: {relative_path} ---\n{file_content}"
        elif content_type == REFERENCE_TYPE.LOGGED_ERROR:
            query += f"\n\n--- Occurred Error ---\n{data}"
        elif content_type == REFERENCE_TYPE.FILL_TEXT:
            fill_text, title = data
            query += f"\n\n--- {title} ---\n{fill_text}"
        elif content_type == REFERENCE_TYPE.CURRENT_FILE:
            relative_path = os.path.relpath(file_path, root_dir)
            query += f"\n\n--- FILE at: {relative_path} ---\n{updated_content}"
    
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
    contents, referenced_contents = extract_content_references_and_comments(
        file_path, root_dir
    )
    with open('test.txt', 'w') as f:
        f.write('This is the contents\n' + contents + '\n')
        f.write('These are the referenced contents\n' + str(referenced_contents) + '\n')
    start_text, updated_content, end_text = process_file(file_path, contents)
    # with open('test.txt', 'a') as f:
    #     f.write('This is the start text\n' + start_text + '\n')
    #     f.write('This is the updated content\n' + updated_content + '\n')
    #     f.write('This is the end text\n' + end_text + '\n')

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
