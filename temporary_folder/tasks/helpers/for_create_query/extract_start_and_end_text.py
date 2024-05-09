import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from temporary_folder.tasks.helpers.for_create_query.line_validation import (
    line_validation_for_start_tag,
    line_validation_for_end_tag,
)


def extract_start_and_end_text(filepath, contents=None):
    """
    Function to extract the text followed by the start and end tags in a file.

    Args:
        filepath (str): The path to the file.
        contents (str): The content of the file.

    Returns:
        start_text (str): The text followed the start tag.
        updated_content (str): The content of the file without the tags.
        end_text (str): The text followed the end tag.
    """
    if contents is None:
        with open(filepath, "r") as file:
            lines = file.readlines()
    else:
        lines = contents.splitlines()
    start_text = ""
    end_text = ""
    updated_lines = []

    for line in lines:
        if result := line_validation_for_start_tag(line):
            start_text = start_text or result
        elif result := line_validation_for_end_tag(line):
            end_text = end_text or result
        else:
            updated_lines.append(line + "\n")
    return start_text, "".join(updated_lines), end_text
