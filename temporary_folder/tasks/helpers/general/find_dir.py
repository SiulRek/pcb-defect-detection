import os


def find_nearest_dir(dir_name, root_dir, reference_dir):
    root_dir = os.path.abspath(root_dir)
    reference_dir = os.path.abspath(reference_dir)
    closest_dir = None
    min_distance = float("inf")

    if dir_name == ".":
        return root_dir
    for dirpath, dirnames, _ in os.walk(root_dir):
        if dir_name in dirnames:
            current_dir = os.path.join(dirpath, dir_name)
            current_relative_path = os.path.relpath(current_dir, root_dir)
            reference_relative_path = os.path.relpath(reference_dir, root_dir)

            current_path_parts = current_relative_path.split(os.sep)
            reference_path_parts = reference_relative_path.split(os.sep)

            # Calculate the 'distance' between current_dir and reference_dir
            distance = len(
                set(current_path_parts).symmetric_difference(set(reference_path_parts))
            )

            if distance < min_distance:
                min_distance = distance
                closest_dir = current_dir

    if not closest_dir:
        msg = f"Directory '{dir_name}' not found near '{reference_dir}'"
        raise FileNotFoundError(msg)

    return closest_dir


def find_dir_from_path_fragment(path_fragment, root_dir):
    root_dir = os.path.abspath(root_dir)
    path_fragment = path_fragment.replace("\\", os.sep).replace("/", os.sep)
    for dirpath, _, _ in os.walk(root_dir):
        if path_fragment in dirpath.lower():
            return dirpath

    msg = f"Directory from fragment '{path_fragment}' not found in '{root_dir}'"
    raise FileNotFoundError(msg)


def find_dir(string, root_dir, reference_dir):
    """
    Function to find the directory from the string. If the string contains a
    path fragment, the function will search for the directory from the path
    fragment. If the string contains only the directory name, the function will
    search for the nearest directory to the reference directory.

    Args:
        - string (str): The string to be searched, which could be a path
            fragment or directory name.
        - root_dir (str): The root directory of the project.
        - reference_dir (str): The path to the reference directory, used
            when finding the nearest directory.

    Returns:
        - dir_path (str): The path to the directory.
    """
    if "\\" in string or "/" in string:
        dir = find_dir_from_path_fragment(string, root_dir)
    else:
        dir = find_nearest_dir(string, root_dir, reference_dir)
    return os.path.normpath(dir)
