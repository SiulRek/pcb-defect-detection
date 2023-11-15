import ast
import re

def parse_and_repeat(input_string):
    # Split the input string by commas, but only those outside of brackets
    elements = re.split(r',\s*(?=[\[\]])', input_string)

    result = []

    for elem in elements:
        # Find the pattern [list]*number
        match = re.match(r'(\[.*\])\*(\d+)', elem)
        if match:
            list_part = ast.literal_eval(match.group(1))
            repeat_count = int(match.group(2))

            # Repeat the elements as required
            for _ in range(repeat_count):
                result.extend(list_part)
        elif elem == '':
            result.extend('')
        else:
            # For elements without repetition, convert and add each item
            list_part = ast.literal_eval(elem)
            result.extend(list_part)

    return result

