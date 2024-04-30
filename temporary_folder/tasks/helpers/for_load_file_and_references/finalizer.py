import os

from temporary_folder.tasks.helpers.for_load_file_and_references.line_validation import (
    line_validation_for_make_query,
    line_validation_for_checksum,
)
from temporary_folder.tasks.helpers.general.make_query import make_query
from temporary_folder.tasks.helpers.general.extract_python_code import extract_python_code

def write_to_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


class Finalizer:
    """
    Class to finalize the process of loading a file and references.
    This includes writing the updated content to the file, writing the query to a file,
    and making a query if required. Additionally, it validates the checksum of the file.
    """

    def __init__(self):
        self.file_path = None
        self.query_path = None
        self.response_path = None

    def set_paths(self, file_path, query_path, response_path):
        self.file_path = file_path
        self.query_path = query_path
        self.response_path = response_path

    def finalize(self, updated_contents, query):
        self.updated_contents = updated_contents
        self.query = query
        final_lines, make_query_flag, max_tokens, create_python_script = self.process_lines()
        self.validate_checksum(final_lines)
        self.write_results(final_lines, query, make_query_flag, max_tokens, create_python_script)

    def process_lines(self):
        final_lines = []
        make_query_flag = False
        max_tokens = None
        create_python_script = False

        for line in self.updated_contents.splitlines():
            if result := line_validation_for_make_query(line.strip()):
                make_query_flag, max_tokens, create_python_script = result
            elif result := line_validation_for_checksum(line.strip()):
                self.checksum = result
            else:
                final_lines.append(line)

        return final_lines, make_query_flag, max_tokens, create_python_script

    def validate_checksum(self, final_lines):
        if self.checksum:
            with open(self.file_path, "r", encoding="utf-8") as file:
                file_contents = file.read().splitlines()
                diff = len(file_contents) - len(final_lines)
                if diff != self.checksum:
                    raise ValueError(f"Checksum mismatch: {diff} != {self.checksum}")

    def write_results(self, final_lines, query, make_query_flag, max_tokens, create_python_script):
        write_to_file(self.file_path, "\n".join(final_lines))
        write_to_file(self.query_path, query)
        if make_query_flag:
            self.handle_query(query, max_tokens, create_python_script)

    def handle_query(self, query, max_tokens, create_python_script):
        print("Making query...")
        response = make_query(query, max_tokens) if max_tokens else make_query(query)
        write_to_file(self.response_path, response)
        print(f"Response saved to {self.response_path}")
        if create_python_script:
            code = extract_python_code(response)
            python_path = os.path.splitext(self.response_path)[0] + ".py"
            write_to_file(python_path, code)
            print(f"Python script saved to {python_path}")