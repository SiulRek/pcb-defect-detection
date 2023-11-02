def recursive_type_conversion(source_value, target_data_type_template):
    """
    Recursively converts the data type of source_value to match the data type of target_data_type_template.
    
    Args:
    - source_value: The value to be converted.
    - param target_data_type_template: A template value whose data type structure is to be replicated.
    
    Return:
    - source_value converted to the data type of target_data_type_template.
    """
    target_type = type(target_data_type_template)

    if target_type in {int, float, str, bool}:
        try:
            return target_type(source_value)
        except ValueError:
            return source_value

    elif target_type is list:
        if isinstance(source_value, list):
            converted_list = []
            for source_item, target_item in zip(source_value, target_data_type_template):
                converted_item = recursive_type_conversion(source_item, target_item)
                converted_list.append(converted_item)
            return converted_list
        else:
            return source_value

    elif target_type is tuple:
        if isinstance(source_value, tuple):
            converted_tuple = tuple(recursive_type_conversion(source_item, target_item) 
                                     for source_item, target_item in zip(source_value, target_data_type_template))
            return converted_tuple
        else:
            return source_value

    elif target_type is dict:
        if isinstance(source_value, dict):
            converted_dict = {}
            for key, target_item in target_data_type_template.items():
                source_item = source_value.get(key)
                converted_dict[key] = recursive_type_conversion(source_item, target_item)
            return converted_dict
        else:
            return source_value

    else:
        return source_value
