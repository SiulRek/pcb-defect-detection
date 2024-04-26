import os


def find_nearest_file(file_name, root_dir, reference_file):
    root_dir = os.path.abspath(root_dir)
    reference_file = os.path.abspath(reference_file)
    closest_file = None
    min_distance = float('inf')

    for dirpath, _, filenames in os.walk(root_dir):
        if file_name in filenames:
            current_file = os.path.join(dirpath, file_name)
            current_relative_path = os.path.relpath(current_file, root_dir)
            reference_relative_path = os.path.relpath(reference_file, root_dir)

            current_path_parts = current_relative_path.split(os.sep)
            reference_path_parts = reference_relative_path.split(os.sep)

            distance = len(set(current_path_parts).symmetric_difference(set(reference_path_parts)))

            if distance < min_distance:
                min_distance = distance
                closest_file = current_file
    if not closest_file:
        raise FileNotFoundError(f"File '{file_name}' not found in '{root_dir}'")
    return closest_file