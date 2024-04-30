from temporary_folder.tasks.helpers.for_load_file_and_references.line_validation import (
    line_validation_for_make_query,
    line_validation_for_checksum,
)
from temporary_folder.tasks.helpers.general.make_query import make_query


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

    def get_line_with_tag(self, tag):
        for line in self.updated_contents.splitlines():
            if tag in line:
                self.query = self.query.replace(line, "")
                self.updated_contents = self.updated_contents.replace(line, "")
                return line
        return None

    def finalize(self, updated_contents, query):
        self.updated_contents = updated_contents
        self.query = query
        final_lines = []
        checksum = None
        make_query_flag = False
        for line in updated_contents.splitlines():
            if result := line_validation_for_make_query(line.strip()):
                make_query_flag, max_tokens = result
            elif result := line_validation_for_checksum(line.strip()):
                checksum = result
            else:
                final_lines.append(line)
        updated_contents = "\n".join(final_lines)
        if checksum:
            with open(self.file_path, "r", encoding="utf-8") as file:
                file_contents = file.read().splitlines()
                diff = len(file_contents) - len(final_lines)
                if diff != checksum:
                    raise ValueError(f"Checksum mismatch: {diff} != {checksum}")
                    
        write_to_file(self.file_path, updated_contents)
        write_to_file(self.query_path, query)
        if make_query_flag:
            print("Making query...")
            if max_tokens:
                response = make_query(query, max_tokens)
            else:
                response = make_query(query)
            response = make_query(query)
            write_to_file(self.response_path, response)
            print(f"Response saved to {self.response_path}")