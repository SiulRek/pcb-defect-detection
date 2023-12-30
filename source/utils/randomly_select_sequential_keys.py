import random
import re

import random
import re


def is_sequential(lst):
    """
    Check if the list elements start from 0 and increment by 1 with each element.

    Args:
    - param lst: List to check.
    - return: Boolean indicating whether the list is sequential starting from 0.
    """
    return all(i == val for i, val in enumerate(lst))


def randomly_select_sequential_keys(input_dict, separator='__'):
    """
    Randomly selects keys from a dictionary that follow a sequential pattern.

    Args:
        input_dict (dict): The input dictionary containing keys to be selected.
        separator (str, optional): The separator used in the key pattern. Defaults to '__'.

    Returns:
        dict: A new dictionary containing randomly selected keys and their corresponding values.
    """

    seq_key_pattern = re.compile(r'.*' + re.escape(separator) + r'L(\d+)')
    freq_key_pattern = re.compile(r'.*' + re.escape(separator) + r'L\d+F(\d+)')   
            
    match_flags = [True if seq_key_pattern.match(key) else False for key in input_dict]

    if not any(match_flags):
        return input_dict
    elif not all(match_flags):
        raise ValueError('Some keys do not match the expected format. TODO change')
    

    indices = list(set([seq_key_pattern.match(key).group(1) for key in input_dict]))
    indices.sort()
    if not is_sequential([int(i) for i in indices]):
        raise ValueError('The indices of the keys are not sequential. TODO Change.')
    
    output_dict = {}
    for index in indices:
        temp_keys = [key for key in input_dict if seq_key_pattern.match(key).group(1) == index]
        keys = []
        for key in temp_keys:
            if freq_key_pattern.match(key):
                freq = int(freq_key_pattern.match(key).group(1))
                keys.extend([key]*freq)
            else:
                keys.append(key)

        selected_key = random.choice(keys)
        output_dict[selected_key] = input_dict[selected_key]

    return output_dict