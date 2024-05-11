import os


def remove_line_comments(text):
    """
    Removes all line comments from a given text. Line comments start with '#'.
    This function handles both full line comments and inline comments.

    Args:
        - text (str): The content of the text from which to remove comments.

    Returns:
        - str: The cleaned content of the text.
    """
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line.startswith("# "):
            clean_line = line.split("# ", 1)[0] + "\n" if "# " in line else line
            new_lines.append(clean_line)
    cleaned_text = "\n".join(new_lines)
    return cleaned_text


def remove_line_comments_from_file(file_path):
    """
    Removes all line comments from a file. Line comments start with '#'.

    Args:
        - file_path (str): The path to the file from which to remove
            comments.
    """
    if not os.path.exists(file_path):
        msg = f"The file {file_path} does not exist."
        raise FileNotFoundError(msg)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    cleaned_text = remove_line_comments(text)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)


if __name__ == "__main__":
    file_path = "path/to/your/file"
    remove_line_comments_from_file(file_path)
