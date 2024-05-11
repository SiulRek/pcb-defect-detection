SEPARATOR = "import"


def is_word(string):
    accepted_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
    for char in string:
        if char not in accepted_chars:
            return False
    return True


def process_name_specifications(name_specifications):
    """
    Processes the name specifications of an import statement. The name
    specifications are a list of strings containing the original name and alias
    name of the imported module. The alias name is the name after the 'as'
    keyword. If the alias name is not provided, the original name is used as the
    alias name.

    Args:
        - name_specifications (str): A string containing the name
            specifications of an import statement.

    Returns:
        - tuple: A tuple containing two lists. The first list contains the
            original names of the importedmodules. The second list contains the
            alias names of the imported modules.
    """
    original_names = []
    alias_names = []

    name_specifications = name_specifications.replace("(", "").replace(")", "")
    parts = name_specifications.split(",")
    for part in parts:
        part = part.strip()
        if " as " in part:
            original_name, alias_name = part.split(" as ")
            original_name = original_name.strip()
            alias_name = alias_name.strip()
            if not is_word(alias_name):
                msg = f"{alias_name} is not a valid alias name."
                raise ValueError(msg)
            if not is_word(original_name):
                msg = f"{original_name} is not a valid original name."
                raise ValueError(msg)
            original_names.append(original_name)
            alias_names.append(alias_name)
        else:
            if not is_word(part):
                msg = f"{part} is not a valid original name."
                raise ValueError(msg)
            original_names.append(part)
            alias_names.append(part)
    return original_names, alias_names


def split_import_statement(import_statement):
    """
    Splits an import statement into its base, original names, and alias names.
    The base is the keyword used to import the module. The original names are
    the names of the modules being imported. The alias names are the names used
    to refer to the imported modules in the code.

    Args:
        - import_statement (str): The import statement to split.

    Returns:
        - tuple: A tuple containing the base, original names, and alias
            names of the import statement.
    """
    words = import_statement.strip().split()
    if "import" not in words:
        msg = f"Import Statement does not contain 'import' keyword:"
        msg += f"{import_statement}"
        raise ValueError(msg)

    if import_statement.startswith("import"):
        base = "import"
        name_specifications = import_statement.split("import ")[1]
    else:
        base = import_statement.split(" import ")[0] + " import"
        name_specifications = import_statement.split(" import ")[1]

    original_names, alias_names = process_name_specifications(name_specifications)

    return base, original_names, alias_names


if __name__ == "__main__":
    import_statements = [
        "import os",
        "import source",
        "import random",
        "import sys",
        "import numpy as np",
        "from math import (\nsqrt,\nceil,\nfloor\n)",
        "from datetime import datetime, timedelta",
        "import sklearn",
        "from pandas.core import series as pd_series",
    ]

    extracted_data = []
    for statement in import_statements:
        extracted_data.append(split_import_statement(statement))
    for data in extracted_data:
        print(data)
