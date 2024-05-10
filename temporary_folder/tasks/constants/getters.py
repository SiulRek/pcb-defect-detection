import os
import json


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_DIR, "..", "..", "..")


def get_response_file_path(root_dir=ROOT_DIR):
    return os.path.join(root_dir, "temporary_folder", "tasks", "outputs", "answer.txt")


def get_temporary_file_path(root_dir=ROOT_DIR):
    return os.path.join(root_dir, "temporary_folder", "tasks", "outputs", "temporary_file.txt")


def get_fill_text_directory(root_dir=ROOT_DIR):
    return os.path.join(root_dir, "temporary_folder", "tasks", "data", "fill_texts")


def get_output_directory(root_dir=ROOT_DIR):
    return os.path.join(root_dir, "temporary_folder", "tasks", "outputs")


def get_checkpoints_directory(root_dir=ROOT_DIR):
    output_dir = get_output_directory(root_dir)
    return os.path.join(output_dir, "checkpoints")


def get_environment_path(root_dir=ROOT_DIR):
    return os.path.join(root_dir, "venv")


def get_modules_info(root_dir=ROOT_DIR):
    path = os.path.join(
        root_dir, "temporary_folder", "tasks", "constants", "modules_info.json"
    )
    with open(path, "r") as file:
        modules_info = json.load(file)
    list_dirs = os.listdir(root_dir)
    modules_info["local"] = [dir for dir in list_dirs if os.path.isdir(dir)]
    return modules_info
