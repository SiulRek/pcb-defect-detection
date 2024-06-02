import random
import re


def is_sequential(lst):
    """ Check if the list elements start from 0 and increment by 1 with each
    element. """
    return all(i == val for i, val in enumerate(lst))


def matches_pattern(key, pattern):
    """ Check if a key matches a given regular expression pattern. """
    return pattern.match(key) is not None


def extract_indices(input_dict, pattern):
    """ Extract indices from keys in the input dictionary based on a given pattern. """
    return [
        int(pattern.match(key).group(2))
        for key in input_dict
        if matches_pattern(key, pattern)
    ]


def filter_keys_by_pattern(input_dict, pattern):
    """ Filter keys in the input dictionary that match a given pattern. """
    return [key for key in input_dict if matches_pattern(key, pattern)]


def remove_second_group(s, pattern):
    """
    Remove the content of the second group from the matched pattern in the given
    string. If the pattern does not match, return the whole input string.

    Args:
        - s (str): The input string to process.
        - pattern (Pattern): The compiled regular expression pattern with at
            least two groups.

    Returns:
        - (str): Modified string with the second group content removed, or
            the original string if no match is found.
    """
    match = pattern.search(s)
    if not match:
        return s

    def replace_with_first_and_third_group(match):
        return match.group(1) + match.group(3)

    return pattern.sub(replace_with_first_and_third_group, s)


def randomly_select_sequential_keys(input_dict, separator="__"):
    """
    Randomly selects keys from a dictionary that follow a specific sequential
    pattern. The pattern is defined by a separator followed by 'I' and a number,
    optionally followed by 'F' and another number. The function selects keys
    based on their indices and frequencies, if specified.

    Args:
        - input_dict (dict): The input dictionary containing keys to be
            selected.
        - separator (str, optional): The separator used in the key pattern,
            defaulting to '__'.

    Returns:
        - (dict): A new dictionary containing randomly selected keys and
            their corresponding values.

    The function first checks if each key matches either the index pattern
    (e.g., '__I1') or the frequency pattern (e.g., '__I1F2'). If all keys match
    one of these patterns, the function then extracts the indices from the keys
    and checks if they form a sequential series starting from 0. Afterward, it
    randomly selects one key per index, considering the frequency of keys where
    specified. The selected keys are then processed to remove the pattern,
    preserving the parts of the key without the specific sequential pattern.
    """

    end_pattern = rf"($|{re.escape(separator)}\S+)"
    ind_key_pattern = re.compile(rf"(.*?){re.escape(separator)}I(\d+)(.*?)")
    ind_key_pattern_all_groups = re.compile(rf"(.*?)({re.escape(separator)}I\d+)(.*?)")
    ind_key_pattern_end = re.compile(rf"(.*?){re.escape(separator)}I(\d+){end_pattern}")
    freq_key_pattern = re.compile(
        rf"(.*?){re.escape(separator)}I\d+F(\d+){end_pattern}"
    )
    freq_key_pattern_all_groups = re.compile(
        rf"(.*?)({re.escape(separator)}I\d+F\d+){end_pattern}"
    )
    match_flags = []
    for key in input_dict:
        match = matches_pattern(key, ind_key_pattern_end) or matches_pattern(
            key, freq_key_pattern
        )
        match_flags.append(match)
    if not any(match_flags):
        return input_dict

    if not all(match_flags):
        msg = "Some keys do not follow a sequential pattern."
        raise KeyError(msg)

    indices = extract_indices(input_dict, ind_key_pattern)
    unique_sorted_indices = sorted(set(indices))
    if not is_sequential(unique_sorted_indices):
        msg = "Indices of the keys are not sequential."
        raise KeyError(msg)

    output_dict = {}
    for index in unique_sorted_indices:
        compiled_pattern = re.compile(
            rf".*{re.escape(separator)}I{index}(F\d+|{end_pattern})"
        )
        temp_keys = filter_keys_by_pattern(input_dict, compiled_pattern)
        keys = []
        for key in temp_keys:
            if match := freq_key_pattern.match(key):
                freq = int(match.group(2))
                keys.extend([key] * freq)
            else:
                keys.append(key)

        selected_key = random.choice(keys)
        new_key = remove_second_group(selected_key, freq_key_pattern_all_groups)
        new_key = remove_second_group(new_key, ind_key_pattern_all_groups)

        if new_key in output_dict:
            msg = "The selected key already exists in the output dictionary."
            raise KeyError(msg)
        output_dict[new_key] = input_dict[selected_key]

    return output_dict
