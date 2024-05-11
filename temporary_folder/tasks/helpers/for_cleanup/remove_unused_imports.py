import os
import re

from temporary_folder.tasks.helpers.for_cleanup.is_specifier_used import (
    is_specifier_used,
)
from temporary_folder.tasks.helpers.for_cleanup.separate_imports import separate_imports
from temporary_folder.tasks.helpers.for_cleanup.split_import_statement import (
    split_import_statement,
)


def reconstruct_specifier_string(specifier):
    original_name, alias_name = specifier
    if original_name == alias_name:
        return original_name
    return f"{original_name} as {alias_name}"


def remove_unused_imports(code_text):
    """
    Removes unused imports from a Python code string.

    Args:
        - code_text (str): The Python code string to remove unused imports
            from.

    Returns:
        - str: The Python code string with unused imports removed.
    """
    updated_code = code_text
    import_statements, code_lines = separate_imports(code_text)
    for statement in import_statements:
        base, original_names, alias_names = split_import_statement(statement)
        specifiers = zip(original_names, alias_names)
        used_specifiers = []
        for specifier in specifiers:
            if is_specifier_used(specifier[1], code_lines, include_imports=True):
                used_specifiers.append(specifier)
        if used_specifiers == []:
            pattern = rf"{statement} *\n"
            updated_code = re.sub(pattern, "", updated_code)
        elif used_specifiers != specifiers:
            used_specifiers = list(map(reconstruct_specifier_string, used_specifiers))
            restored_import_statement = base + " " + ", ".join(used_specifiers)
            updated_code = updated_code.replace(statement, restored_import_statement)
    return updated_code


def remove_unused_imports_from_file(file_path):
    """
    Removes unused imports from a Python file.

    Args:
        - file_path (str): The path to the Python file to remove unused
            imports from.
    """
    if not os.path.exists(file_path):
        msg = f"The file {file_path} does not exist."
        raise FileNotFoundError(msg)
    with open(file_path, "r") as file:
        code_text = file.read()
    updated_code = remove_unused_imports(code_text)
    with open(file_path, "w") as file:
        file.write(updated_code)


if __name__ == "__main__":
    path = r"path/to/file.py"
    remove_unused_imports_from_file(path)
    print(f"Unused imports removed from {path}.")
