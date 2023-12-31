import random
import re


def is_sequential(lst):
    """Check if the list elements start from 0 and increment by 1 with each element."""
    return all(i == val for i, val in enumerate(lst))


def matches_pattern(key, pattern):
    """Check if a key matches a given regular expression pattern."""
    return pattern.match(key) is not None


def extract_indices(input_dict, pattern):
    """Extract indices from keys in the input dictionary based on a given pattern."""
    return [int(pattern.match(key).group(1)) for key in input_dict if matches_pattern(key, pattern)]


def filter_keys_by_pattern(input_dict, pattern):
    """Filter keys in the input dictionary that match a given pattern."""
    return [key for key in input_dict if matches_pattern(key, pattern)]


def randomly_select_sequential_keys(input_dict, separator='__'):
    """
    Randomly selects keys from a dictionary that follow a sequential pattern.

    Args:
    - input_dict (dict): The input dictionary containing keys to be selected.
    - separator (str, optional): The separator used in the key pattern. Defaults to '__'.

    Returns:
    - (dict): A new dictionary containing randomly selected keys and their corresponding values.
    """
    seq_key_pattern = re.compile(rf'.*{re.escape(separator)}L(\d+)')
    seq_key_pattern_end = re.compile(rf'.*{re.escape(separator)}L(\d+)$')
    freq_key_pattern = re.compile(rf'.*{re.escape(separator)}L\d+F(\d+)$')

    match_flags = [matches_pattern(key, seq_key_pattern_end) or matches_pattern(key, freq_key_pattern) for key in input_dict]
    if not any(match_flags):
        return input_dict

    if not all(match_flags):
        raise KeyError('Some keys do not follow a sequential pattern.')

    indices = extract_indices(input_dict, seq_key_pattern)
    unique_sorted_indices = sorted(set(indices))
    if not is_sequential(unique_sorted_indices):
        raise KeyError('Indices of the keys are not sequential.')

    output_dict = {}
    for index in unique_sorted_indices:
        temp_keys = filter_keys_by_pattern(input_dict, re.compile(rf'.*{re.escape(separator)}L{index}(F\d+|$)'))
        keys = []
        for key in temp_keys:
            if match := freq_key_pattern.match(key):
                freq = int(match.group(1))
                keys.extend([key] * freq)
            else:
                keys.append(key)

        selected_key = random.choice(keys)
        output_dict[selected_key] = input_dict[selected_key]

    return output_dict
