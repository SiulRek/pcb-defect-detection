import os


def find_file_from_path_fragment(path_fragment, root_dir):
    """
    Searches for a file that matches a path fragment within the specified root directory.

    Args:
        path_fragment (str): A partial path or filename to search for.
        root_dir (str): The root directory to search within.

    Returns:
        str: The full path to the first file found that matches the path fragment.
        Returns None if no matching file is found.
    """
    root_dir = os.path.abspath(root_dir)
    path_fragment = path_fragment.lower()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename).lower()
            if path_fragment in full_path:
                return os.path.join(dirpath, filename)

    return None
