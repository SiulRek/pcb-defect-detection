import re


ROUND_BRACKET_PATTERN = re.compile(r"\((.*?)\)")
SQUARE_BRACKET_PATTERN = re.compile(r"\[(.*?)\]")

def retrieve_list_in_square_brackets(string, error_message):
    """ Get the list in square brackets."""
    match = SQUARE_BRACKET_PATTERN.match(string)
    if not match:
        raise ValueError(error_message)
    out_list = match.group(1).split(";")
    out_list = [elem.strip() for elem in out_list]
    out_list = '' if (len(out_list) == 1 and out_list[0] == '') else out_list
    return out_list


def retrieve_optional_arguments(string):
    """ Get optional arguments from a string."""
    result = re.search(ROUND_BRACKET_PATTERN, string)
    if result:
        arguments = result.group(1).split(",")
        arguments = [argument.strip() for argument in arguments]
        return arguments
    return None


def retrieve_bool(string):
    """ Get a boolean value from a string."""
    return True if string.strip().lower() == "true" else False
