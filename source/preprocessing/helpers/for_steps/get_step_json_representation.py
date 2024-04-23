def format_parameters(parameters):
    """Formats the parameters dictionary into a string that can be added to a JSON file."""
    formatted_items = []
    for k, v in parameters.items():
        if isinstance(v, str):
            formatted_value = f'"{v}"'
        else:
            formatted_value = str(v).replace("True", "true").replace("False", "false")
        formatted_item = f'        "{k}": {formatted_value}'
        formatted_items.append(formatted_item)

    parameters_str = ",\n".join(formatted_items)
    return parameters_str


def get_step_json_representation(parameters, step_name):
    """
    Returns strings that corresponds to JSON entry text of the preprocessing step.

    Args:
        parameters (dict): The parameters of the preprocessing step.
        step_name (str): The name of the preprocessing step.

    Returns:
        str: The JSON representation of the preprocessing step.
    """
    # Convert datatype of values of parameters to match JSON format
    conv_parameters = {}
    for key, value in parameters.items():
        if isinstance(value, tuple):
            value = list(value)
        conv_parameters[key] = value

    parameters_str = format_parameters(conv_parameters)
    json_string = f'    "{step_name}": {{\n{parameters_str}\n    }}'
    return json_string
