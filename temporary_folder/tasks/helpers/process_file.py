import os
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from temporary_folder.tasks.constants.definitions import START_TAG, END_TAG, ERROR_TAG

def process_file(filepath, contents=None):
    """ 
    Function to process the file and extract the start_text, end_text and error_text

    Args:
        filepath (str): The path to the file.
        contents (str): The content of the file.
    
    Returns:
        start_text (str): The text between the start tag.
        updated_content (str): The content of the file without the tags.
        end_text (str): The text between the end tag.
    """
    if contents is None:
        with open(filepath, "r") as file:
            lines = file.readlines()
    else:
        lines = contents.splitlines()
    start_text = ''
    end_text = ''
    error_text = ''
    updated_lines = []

    for line in lines:
        if START_TAG in line and start_text == '':
            start_text = line.split(START_TAG, 1)[1].strip()
        elif END_TAG in line and end_text == '':
            end_text = line.split(END_TAG, 1)[1].strip()
        elif ERROR_TAG in line and error_text == '':
            error_text = get_error_text(filepath)
        else:
            updated_lines.append(line + '\n')

    with open(filepath, "w") as file:
        file.writelines(updated_lines)
    if error_text:
        end_text += f"\n\n---Occurred Error:---{error_text}"
    return start_text, ''.join(updated_lines), end_text

## ------- Functionality to extract Errors from logger
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

def extract_error_messages(log_text):
    message_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (ERROR|INFO) - .*)')
    
    lines = log_text.split('\n')
    messages = ['']
    for line in lines:
        if message_pattern.match(line):
            messages.append(line)
        else:
            messages[-1] += '\n' + line

    captured_messages = []
    for message in messages:
        if not 'Test passed' in message:
            captured_messages.append(message)
    if not captured_messages:
        raise ValueError("No error messages found in log text.")
    return '\n'.join(captured_messages)

def get_error_text(file_path):
    log_path = find_nearest_file('test_results.log', ROOT_DIR, file_path)
    with open(log_path) as f:
        log_text = f.read()
    return extract_error_messages(log_text)