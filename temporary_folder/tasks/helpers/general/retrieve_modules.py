import json

import pkg_resources
import stdlib_list

package_to_import_name = {
    "Pillow": "PIL",
    # 'scikit-learn': 'sklearn',
    "PyYAML": "yaml",
    "python-dateutil": "dateutil",
    # Add other packages where needed.
}


def retrieve_modules(json_file):
    """
    Retrieves names of standard library modules and third-party modules in the
    current Python environment. Stores them in a JSON file under
    'standard_library' and 'third_party' keys.

    Args:
        - json_file (str): The path to the JSON file to store the modules
            information.
    """
    standard_libs = stdlib_list.stdlib_list()
    standard_libs = [lib for lib in standard_libs if "." not in lib]
    for i, lib in enumerate(standard_libs):
        if lib in package_to_import_name:
            standard_libs[i] = package_to_import_name[lib]

    third_party_libs = [dist.project_name for dist in pkg_resources.working_set]
    third_party_libs = [lib for lib in third_party_libs if "." not in lib]
    for i, lib in enumerate(third_party_libs):
        if lib in package_to_import_name:
            third_party_libs[i] = package_to_import_name[lib]

    modules_dict = {"standard_library": standard_libs, "third_party": third_party_libs}

    with open(json_file, "w") as file:
        json.dump(modules_dict, file, indent=4)

    print(f"Modules information stored in {json_file}")


if __name__ == "__main__":
    json_file = "./temporary_folder/tasks/constants/modules_info.json"
    retrieve_modules(json_file)
