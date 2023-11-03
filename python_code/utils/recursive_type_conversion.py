def recursive_type_conversion(source_value, value_of_target_datatype):
    """
    Converts the data type of a given value to match that of a template value, doing so recursively for complex data structures.

    This function aims to transform the data type of `source_value` so that it matches the data type of `value_of_target_datatype`. 
    The conversion is performed recursively, which means that if `source_value` is a collection (e.g., list, tuple, dictionary), 
    each item within the collection will also be converted to match the corresponding item in the `value_of_target_datatype`.

    Args:
    - source_value: The value whose data type is to be converted. It can be of any type: primitive data types, list, tuple, or dict.
    - value_of_target_datatype: An instance of the desired data type that serves as a template for conversion. 
                                It should be of the same structure as the `source_value` if it is a collection.

    Returns:
    - The `source_value` converted to the data type structure of `value_of_target_datatype`.
    """
    
    target_type = type(value_of_target_datatype)

    if target_type in {int, float, str, bool}:
        try:
            return target_type(source_value)
        except ValueError:
            return source_value

    elif target_type is list:
        if isinstance(source_value, tuple):
            source_value = list(source_value)
        elif not isinstance(source_value, list):
            raise TypeError(f"Value '{source_value}' cannot be recursivly converted to {value_of_target_datatype}.")
        converted_list = []
        for source_item, target_item in zip(source_value, value_of_target_datatype):
            converted_item = recursive_type_conversion(source_item, target_item)
            converted_list.append(converted_item)
        return converted_list

    elif target_type is tuple:
        if isinstance(source_value, list):
            source_value = tuple(source_value)
        elif not isinstance(source_value, tuple):
            raise TypeError(f"Value '{source_value}' cannot be recursivly converted to {value_of_target_datatype}.")
        converted_list = []
        for source_item, target_item in zip(source_value, value_of_target_datatype):
            converted_item = recursive_type_conversion(source_item, target_item)
            converted_list.append(converted_item)
        return tuple(converted_list)

    elif target_type is dict:
        if isinstance(source_value, dict):
            converted_dict = {}
            for key, target_item in value_of_target_datatype.items():
                source_item = source_value.get(key)
                converted_dict[key] = recursive_type_conversion(source_item, target_item)
            return converted_dict
        else:
            return source_value

    else:
        return source_value

if __name__ == '__main__':
    # source_value = [10,10]
    # value_of_target_datatype = (10,10)
    # print(recursive_type_conversion(source_value, value_of_target_datatype))
    print(int('10'),'+', int('20'), '=', float('10') + float('20'))
