import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from temporary_folder.tasks.constants.definitions import START_TAG, END_TAG


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
        if START_TAG in line and start_text == "":
            start_text = line.split(START_TAG, 1)[1].strip()
        elif END_TAG in line and end_text == "":
            end_text = line.split(END_TAG, 1)[1].strip()
        else:
            updated_lines.append(line + "\n")

    with open(filepath, "w") as file:
        file.writelines(updated_lines)
    return start_text, "".join(updated_lines), end_text
