import os
from temporary_folder.tasks.constants.getters import get_fill_text_directory 


def get_fill_text(placeholder, root_dir):
    """Get the fill text from the fill text directory."""
    fill_text_directory = get_fill_text_directory(root_dir)
    for root, _, files in os.walk(fill_text_directory):
        files = [os.path.splitext(file) for file in files]
        for file in files:
            if file[0] == placeholder:
                title_parts = os.path.basename(root).split("_")
                title = " ".join([word.capitalize() for word in title_parts])
                file_path = os.path.join(root, f"{file[0]}{file[1]}")
                with open(file_path, "r", encoding="utf-8") as fill_text_file:
                    fill_text = fill_text_file.read()
                return fill_text, title
    raise FileNotFoundError(f"Fill text with placeholder {placeholder} not found.")