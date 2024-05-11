import os
import json

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_DIR, "..", "..", "..")


def get_response_file_path(root_dir=ROOT_DIR):
    path = os.path.join(root_dir, "temporary_folder", "tasks", "outputs", "answer.txt")
    return os.path.normpath(path)


def get_temporary_file_path(root_dir=ROOT_DIR):
    path = os.path.join(root_dir, "temporary_folder", "tasks", "outputs", "temporary_file.txt")
    return os.path.normpath(path)


def get_fill_text_directory(root_dir=ROOT_DIR):
    path = os.path.join(root_dir, "temporary_folder", "tasks", "data", "fill_texts")
    return os.path.normpath(path)


def get_query_templates_directory(root_dir=ROOT_DIR):
    path = os.path.join(root_dir, "temporary_folder", "tasks", "data", "query_templates")
    return os.path.normpath(path)


def get_output_directory(root_dir=ROOT_DIR):
    path = os.path.join(root_dir, "temporary_folder", "tasks", "outputs")
    return os.path.normpath(path)


def get_checkpoint_directory(root_dir=ROOT_DIR):
    output_dir = get_output_directory(root_dir)
    path = os.path.join(output_dir, "checkpoints")
    return os.path.normpath(path)


def get_environment_path(root_dir=ROOT_DIR):
    path = os.path.join(root_dir, "venv")
    return os.path.normpath(path)


def get_modules_info(root_dir=ROOT_DIR):
    path = os.path.join(
        root_dir, "temporary_folder", "tasks", "constants", "modules_info.json"
    )
    with open(path, "r") as file:
        modules_info = json.load(file)
    list_dirs = os.listdir(root_dir)
    modules_info["local"] = [dir for dir in list_dirs if os.path.isdir(dir)]
    return modules_info