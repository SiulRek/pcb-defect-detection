import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def search_files_with_name(root_path, filename):
    """
    Search for files with an exact name in a specified directory and its
    subdirectories.

    Args:
        - root_path (str): The root directory path to start the search from.
        - filename (str): The exact name of the file to search for.

    Returns:
        - list: A list of paths to the files that match the specified
            filename.
    """
    matches = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file == filename:
                matches.append(os.path.join(root, file))
    return matches


if __name__ == "__main__":

    i = 1
    for file_path in search_files_with_name(ROOT_DIR, "test_runner.py"):
        print(i, ": ", file_path)
        i += 1
