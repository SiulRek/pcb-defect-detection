import os
import re
import json
import random

from python_code.utils import recursive_type_conversion

class ConfigurationHandler:

    KEY_SEPARATOR = '__'       

    def __init__(self, json_path, instance_mapping): 
        self.json_path = json_path
        self.instance_mapping = instance_mapping
    
    @property
    def json_path(self):
        return self._json_path
    
    @json_path.setter
    def json_path(self, value):
        if not os.path.exists(value):
            raise ValueError(f"The specified JSON file does not exist: {value}")
        if not value.endswith('.json'):
            raise ValueError(f"The specified file is not JSON: {value}")
        self._json_path = value

    @property
    def instance_mapping(self):
        return self._instance_mapping
    
    @instance_mapping.setter
    def instance_mapping(self, value):
        if not isinstance(value, dict):
            raise ValueError(f"The specified instance mapping is not of type dict: {value}.")
        self._instance_mapping = value

    def save_instance_list_to_json(self, instance_list):
        configs = {}
        for instance in instance_list:
            configs = self._add_instance_to_configs(instance, configs) 
        
        self.save_configs_to_json(configs)
    
    def _add_instance_to_configs(self, instance, configs):
        for class_name, mapped_class in self.instance_mapping.items():
            if isinstance(instance, mapped_class):
                class_name = self._generate_unique_key_name(class_name, configs)
                if hasattr(instance, 'params'):
                    configs[class_name] = instance.params
                    return configs
                else:
                    raise AttributeError(f"Mapped class: '{mapped_class}' does not have the attribute 'params'.")
        raise KeyError(f"Instance '{instance}' is not a value in 'instance_mapping' dict.")
    
    def save_configs_to_json(self, configs): 
        # Convert the configs (dict of dict) to a json writebla type, make values to element of a list -> [values], handle duplications, improve visible appearance of string and write to json.

        json_data = {}
        for class_name in configs.keys():

            converted_params = {}
            for key, value in configs[class_name].items():
                converted_params[key] = [self._serialize_to_json_value(value)] # Square Brackets required as parameter ranges are expected instead of values.
        
            unique_name = self._generate_unique_key_name(class_name, json_data)   
            json_data[unique_name] = converted_params

        # Write 'json_data' to JSON file
        json_string = json.dumps(json_data, indent=4)
        json_string = json_string.replace('},', '},\n')
        pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'   
        result = re.sub(pattern, self._remove_newlines, json_string)  # Replace newlines and spaces within square brackets (improves readability).
        with open(self.json_path, 'w') as file:
            file.write(result)
    
    def _serialize_to_json_value(self, obj):
        """ Helper method to recursively convert values to a accepted JSON format."""
        if isinstance(obj, tuple) or isinstance(obj, list):
            return [self._serialize_to_json_value(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._serialize_to_json_value(value) for key, value in obj.items()}
        elif type(obj) in {int, float, str, bool}:
            return obj
        else:
            raise TypeError(f"Object with value '{obj} cannot not be serialized to JSON format.")
        
    def _generate_unique_key_name(self, current_key, dictionary):        
        """ Generates a unique step name for JSON entries to avoid conflicts."""
        key = current_key
        i = 2
        while key in dictionary.keys():              # Same namining of entries are not allowed in json.
            key = key.split(ConfigurationHandler.KEY_SEPARATOR)[0] + ConfigurationHandler.KEY_SEPARATOR + str(i)
            i += 1

        return key
    
    def _remove_newlines(self, match):
        """ Removes newlines and spaces within square brackets in JSON strings."""
        return match.group().replace('\n', '').replace(' ', '')    
        
    def get_instance_list_from_json(self, additional_init_params=None): 
        # Get keys from json, mapp keys to classes, convert values related to the keys recursivlyn (optional when class attribute 'init_params_datatypes' exists), call classes with related values, return list of instances.
        with open(self.json_path, 'r') as file:
            json_data = json.load(file)
            class_names = list(json_data.keys())
        
        instance_list = []
        for class_name in class_names:
            instance = self._create_class_instance(class_name, json_data, additional_init_params)
            instance_list.append(instance)

        return instance_list
    
    def _create_class_instance(self, class_name, json_data, additional_init_params=None):
        class_name_parts = class_name.split(ConfigurationHandler.KEY_SEPARATOR)

        if class_name_parts[0] not in self.instance_mapping.keys():
            raise KeyError(f"Class Name '{class_name_parts[0]}' from json file has no instance mapping.")
        mapped_class = self.instance_mapping[class_name_parts[0]]
        init_params = json_data.get(class_name)
        
        if hasattr(mapped_class, 'init_params_datatypes'):

            if mapped_class.init_params_datatypes == None:
                raise AttributeError(f"The class attribute 'init_params_datatypes' of class {mapped_class} cannot be None to create class instance.")
                
            init_params_datatypes = mapped_class.init_params_datatypes
            init_params = {init_param: random.choice(range) for init_param, range in init_params.items()}
            init_params = recursive_type_conversion(init_params, init_params_datatypes)

        try:
            if additional_init_params:
                return mapped_class(**init_params, **additional_init_params)
            else: 
                return mapped_class(**init_params)
        except ValueError as e:
            raise ValueError(f"Incorrect initialization of class {mapped_class}, probably initialization parameters mismatch with JSON file.") from e
        except TypeError as e:
            raise ValueError(f"Incorrect initialization of class {mapped_class}, probably initialization parameters mismatch with JSON file.") from e
