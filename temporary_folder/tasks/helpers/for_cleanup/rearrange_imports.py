import warnings

from temporary_folder.tasks.constants.getters import get_modules_info
from temporary_folder.tasks.helpers.for_cleanup.extract_module_path import (
    extract_module_path,
)
from temporary_folder.tasks.helpers.for_cleanup.separate_imports import separate_imports


def extract_module_docstring(code_text):
    """
    Extracts the module-level docstring from a given string of Python code using
    regular expressions.

    Args:
        - code_text (str): A string containing Python code.

    Returns:
        - str or None: The module-level docstring, if present, otherwise
            None.
    """
    docstring_lines = []
    if code_text.startswith("'''") or code_text.startswith('"""'):
        line_iterator = iter(code_text.splitlines())
        first_line = next(line_iterator)
        docstring_lines.append(first_line)
        stripped_line = first_line.strip()[3:]
        try:
            while not stripped_line.endswith("'''") and not stripped_line.endswith(
                '"""'
            ):
                line = next(line_iterator)
                stripped_line = line.strip()
                docstring_lines.append(line)
        except StopIteration:
            msg = "Module Docstring not closed properly."
            raise ValueError(msg)
        if (
            next(line_iterator).strip() == ""
        ):  # \ Add the empty line after the docstring.
            docstring_lines.append("")
    docstring = "\n".join(docstring_lines) + "\n" if docstring_lines else None
    return docstring


def process_import_statements(import_statements, modules_info_getter=get_modules_info):
    """
    Rearranges the import staments in alphabetical order.

    Args:
        - import_statements (list): A list of import lines.
        - modules_info_getter (function): A function that returns a
            dictionary containing informationabout Python modules. Defaults to
            get_modules_info.

    Returns:
        - list: A list of import lines
    """
    modules_info = modules_info_getter()
    standard_library = modules_info["standard_library"]
    third_party = modules_info["third_party"]
    local = modules_info["local"]
    module_paths = [extract_module_path(line) for line in import_statements]
    package_names = [path.split(".")[0] for path in module_paths]
    import_statements = zip(import_statements, module_paths, package_names)

    import_statements = sorted(import_statements, key=lambda item: item[1])

    standard_library_imports = []
    third_party_imports = []
    local_imports = []
    for import_statement in import_statements:
        if import_statement[2] in standard_library:
            standard_library_imports.append(import_statement)
        elif import_statement[2] in third_party:
            third_party_imports.append(import_statement)
        elif import_statement[2] in local:
            local_imports.append(import_statement)
        else:
            local_imports.append(import_statement)
            msg = f"Warning: '{import_statement[2]}' does not appear in loaded modules information."
            warnings.warn(msg)

    updated_import_statements = []
    if standard_library_imports != []:
        standard_library_imports = [item[0] for item in standard_library_imports]
        updated_import_statements.extend(standard_library_imports)
        updated_import_statements.append("")
    if third_party_imports != []:
        third_party_imports = [item[0] for item in third_party_imports]
        updated_import_statements.extend(third_party_imports)
        updated_import_statements.append("")
    if local_imports != []:
        local_imports = [item[0] for item in local_imports]
        updated_import_statements.extend(local_imports)
        updated_import_statements.append("")
    return updated_import_statements


def rearrange_imports(code_text):
    """
    Rearranges the import statements in a Python script.

    Args:
        - code_text (str): A string containing Python code.

    Returns:
        - str: The Python code with the import statements rearranged.
    """
    import_statements, other_code = separate_imports(code_text)
    updated_import_statements = process_import_statements(import_statements)
    updated_import_statements = "\n".join(updated_import_statements)
    module_docstring = extract_module_docstring(code_text)
    if module_docstring:
        other_code = other_code.replace(module_docstring, "")
        return module_docstring + "\n" + updated_import_statements + "\n" + other_code
    return updated_import_statements + other_code


def rearrange_imports_from_file(file_path):
    """
    Rearranges the import statements in a Python script file.

    Args:
        - file_path (str): The path to the Python script file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        code_text = file.read()

    updated_code = rearrange_imports(code_text)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_code)


if __name__ == "__main__":
    file_path = r"temporary_folder/tasks/tests/data/example_script_3.py"
    rearrange_imports_from_file(file_path)
    print(file_path)
