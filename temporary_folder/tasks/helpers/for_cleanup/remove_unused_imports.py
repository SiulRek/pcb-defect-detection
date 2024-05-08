import os
#TODO Think about how to handle the case where the import statement is split across multiple lines and with indentation
from temporary_folder.tasks.helpers.for_cleanup.is_specifier_used import is_specifier_used
from temporary_folder.tasks.helpers.for_cleanup.split_import_statement import split_import_statement
from temporary_folder.tasks.helpers.for_cleanup.separate_imports import separate_imports


def remove_unused_imports(code_text):
    updated_code = code_text
    import_statements, code_lines = separate_imports(code_text)
    for statement in import_statements:
        base, specifiers = split_import_statement(statement)
        used_specifiers = []
        for specifier in specifiers:
            if is_specifier_used(specifier, code_lines, include_imports=True):
                used_specifiers.append(specifier)
        if used_specifiers == []:
            updated_code = updated_code.replace(statement + '\n', "")
            updated_code = updated_code.replace(statement, "")
        elif used_specifiers != specifiers:
            restored_import_statement = base + " " + ", ".join(used_specifiers)
            updated_code = updated_code.replace(statement, restored_import_statement)
    return updated_code


def remove_unused_imports_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as file:
        code_text = file.read()
    updated_code = remove_unused_imports(code_text)
    with open(file_path, 'w') as file:
        file.write(updated_code)

if __name__ == "__main__":
    path = r"./temporary_folder/tasks/tests/data/example_script_5.py"
    remove_unused_imports_from_file(path)