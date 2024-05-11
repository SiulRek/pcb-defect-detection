import os

from directory_tree import display_tree


def generate_directory_tree(
    path, max_depth=float("inf"), include_files=True, additional_ignore_list=None
):
    """
    Generates a directory structure for the specified path.

    Args:
        - path (str): The path to the directory.
        - max_depth (int): The maximum depth of the directory structure to
            display.Defaults to float('inf').
        - include_files (bool): Whether to include files in the directory
            structure.Defaults to True.
        - additional_ignore_list (list): A list of file or directory names
            to ignore.Defaults to None.

    Returns:
        - str: The directory structure.
    """
    ignore_list = []
    ignore_list.extend(additional_ignore_list or [])
    tree = display_tree(
        path, max_depth=max_depth, string_rep=True, ignore_list=ignore_list
    )
    if not include_files:
        lines = [line for line in tree.split(os.linesep) if line.endswith("/")]
        tree = os.linesep.join(lines)
    return tree


if __name__ == "__main__":
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    print(generate_directory_tree(path, max_depth=1, include_files=False))
    print(generate_directory_tree(path, max_depth=2, include_files=True))
    print(generate_directory_tree(path, max_depth=-1, include_files=True))
