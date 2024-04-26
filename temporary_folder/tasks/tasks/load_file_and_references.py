import re
import sys
import os

ROOT_DIR = sys.argv[1]
FILE_PATH = sys.argv[2]
sys.path.append(ROOT_DIR)
from temporary_folder.tasks.helpers.process_file import process_file
from temporary_folder.tasks.helpers.extract_references_and_contents import extract_references_and_content
import temporary_folder.tasks.helpers.print_statements as task_prints
from temporary_folder.tasks.helpers.add_text_tags import add_text_tags
from temporary_folder.tasks.constants.getters import get_temporary_file_path

TEMPORARY_FILE = get_temporary_file_path(ROOT_DIR)


def load_file_and_references(file_path, root_dir, query_path):
    """
    Load the content of the specified Python file and additionally load any files
    referenced on lines with a '#' symbol.

    Args:
        file_path (str): The path to the Python file.
        root_dir (str): The root directory of the project.
        query_path (str): The path to the file where the query will be saved.
    """
    referenced_files, contents = extract_references_and_content(file_path, root_dir)
    start_text, updated_content, end_text = process_file(file_path, contents)

    query = "--- Original Python File ---\n" + updated_content
    for path, content in referenced_files.items():
        if path.endswith("test.py"):
            query += f"\n\n--- Test File ({path}) ---\n{content}"
        else:
            query += f"\n\n--- Additional File ({path}) ---\n{content}"

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
