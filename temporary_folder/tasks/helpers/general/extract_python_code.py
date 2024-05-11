import re


def extract_python_code(text):
    pattern = r"```(.*?)```"

    matches = re.findall(pattern, text, re.DOTALL)

    python_code_blocks = [
        match
        for match in matches
        if "python" in match.lower()
        or match.strip().startswith(("def", "import", "from", "class"))
    ]
    python_code = "\n\n".join(python_code_blocks)
    if python_code.startswith("python"):
        python_code = python_code[6:]
    python_code = python_code.replace("`", "")
    return python_code


if __name__ == "__main__":
    text = """
Here's an example of a Python function:

```python
def hello_world():
    print("Hello, world!")
#This function prints a greeting.```
Blablabla cool talking
```python
def bye_world():
    print("Bye, world!")
#This function prints a greeting.```
"""

    print(extract_python_code(text))