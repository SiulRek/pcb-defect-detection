import ast
import textwrap


def wrap_docstrings(filepath, max_line_length=72):
    with open(filepath, "r") as file:
        source = file.read()

    wrapper = textwrap.TextWrapper(
        width=max_line_length, replace_whitespace=False, drop_whitespace=False
    )

    class DocstringWrapper(ast.NodeTransformer):
        def __init__(self, source):
            self.modified_source = source

        def generic_visit(self, node):
            nonlocal wrapper
            if hasattr(node, "body") and ast.get_docstring(node, clean=False):
                original_docstring = ast.get_docstring(node, clean=False)
                indented_lines = [
                    "    " + line for line in original_docstring.split("\n")
                ]
                wrapped_docstring = "\n".join(wrapper.wrap(" ".join(indented_lines)))
                modified_docstring = '"""' + wrapped_docstring + '"""'
                start_index = self.modified_source.find(original_docstring)
                if start_index != -1:
                    self.modified_source = (
                        self.modified_source[:start_index]
                        + modified_docstring
                        + self.modified_source[start_index + len(original_docstring) :]
                    )
            return ast.NodeTransformer.generic_visit(self, node)

    wrapper_instance = DocstringWrapper(source)
    wrapper_instance.visit(tree)
    modified_source = wrapper_instance.modified_source

    DocstringWrapper().visit(tree)

    with open("temp.py", "w") as file:
        file.write(modified_source)


if __name__ == "__main__":
    wrap_docstrings("new_test_file.py", max_line_length=5)
