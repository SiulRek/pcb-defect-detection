def extract_module_path(import_statement):
    """
    Extracts the module name from an import statement.

    This function parses a line of Python code that starts with 'import' or
    'from' and extracts the first module or package name mentioned immediately
    after these keywords. The function handles different import styles and
    returns the full name of the module or package as specified in the import
    statement.

    Args:
        - import_statement (str): A string containing a single line of a
            Python import statement.

    Returns:
        - str: The name of the module or package imported.
    """
    stripped_line = import_statement.strip()

    if stripped_line.startswith("from"):
        parts = stripped_line.split()
        if len(parts) > 1:
            return parts[1]

    elif stripped_line.startswith("import"):
        parts = stripped_line.split()
        if len(parts) > 1:
            module_part = parts[1].split(" as ")[0]
            if module_part.endswith(","):
                return module_part[:-1]
            return module_part
    return ""


if __name__ == "__main__":
    print(extract_module_path("import os"))
    print(extract_module_path("from functions.print_text import print_text_function"))
    print(extract_module_path("import tensorflow.keras.Models as models"))
