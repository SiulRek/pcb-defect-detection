def recursive_type_conversion(source_value, target_datatype_template):
    """
    Converts the data type of a given value to match that of a template value, doing so recursively for complex data structures.

    This function aims to transform the data type of `source_value` so that it matches the data type of `target_datatype_template`. 
    The conversion is performed recursively, which means that if `source_value` is a collection (e.g., list, tuple, dictionary), 
    each item within the collection will also be converted to match the corresponding item in the `target_datatype_template`.

    Args:
    - source_value: The value whose data type is to be converted. It can be of any type: primitive data types, list, tuple, or dict.
    - target_datatype_template: An instance of the desired data type that serves as a template for conversion. 
                                It should be of the same structure as the `source_value` if it is a collection. If it is a dictionary `target_datatype_template` shall 
                                have all keys of `source_value`, the other way around is not required.                               

    Returns:
    - The `source_value` converted to the data type structure of `target_datatype_template`.
    """

    target_datatype = type(target_datatype_template)
    
    if target_datatype_template in [int, float, str, bool]:
        try:
            return target_datatype_template(source_value)
        except  ValueError:
            raise TypeError(f"Value '{source_value}' cannot be recursivly converted to {target_datatype_template}.")

    elif target_datatype is list:
        if isinstance(source_value, tuple):
            source_value = list(source_value)
        elif not isinstance(source_value, list):
            raise TypeError(f"Value '{source_value}' cannot be recursivly converted to {target_datatype_template}.")
        converted_list = []
        for source_item, target_item in zip(source_value, target_datatype_template):
            converted_item = recursive_type_conversion(source_item, target_item)
            converted_list.append(converted_item)
        return converted_list

    elif target_datatype is tuple:
        if isinstance(source_value, list):
            source_value = tuple(source_value)
        elif not isinstance(source_value, tuple):
            raise TypeError(f"Value '{source_value}' cannot be recursivly converted to {target_datatype_template}.")
        converted_list = []
        for source_item, target_item in zip(source_value, target_datatype_template):
            converted_item = recursive_type_conversion(source_item, target_item)
            converted_list.append(converted_item)
        return tuple(converted_list)

    elif target_datatype is dict:
        if isinstance(source_value, dict):
            converted_dict = {}
            for key, target_item in target_datatype_template.items():
                source_item = source_value.get(key)
                if source_item is not None:
                    converted_dict[key] = recursive_type_conversion(source_item, target_item)
            return converted_dict
        else:
            return source_value

    else:
        return source_value

if __name__ == '__main__':
    source_value = [10,1]
    target_datatype_template = (int,bool)
    print(recursive_type_conversion(source_value, target_datatype_template))
    print(int('10'),'+', int('20'), '=', float('10') + float('20'))
