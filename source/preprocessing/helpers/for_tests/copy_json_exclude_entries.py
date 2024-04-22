import json

def copy_json_exclude_entries(source_file, dest_file, exclude_keys):
    """
    Copies data from a source JSON file to a destination JSON file,
    excluding certain entries.

    Args:
    - source_file: Path to the source JSON file.
    - dest_file: Path to the destination JSON file.
    - exclude_keys: A list of keys to exclude from copying.
    """
    with open(source_file, 'r') as file:
        data = json.load(file)

    for key in exclude_keys:
        data.pop(key, None)

    with open(dest_file, 'w') as file:
        json.dump(data, file, indent=4)
