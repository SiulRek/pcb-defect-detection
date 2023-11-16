import os
import re
import json
import random

from python_code.utils.recursive_type_conversion import recursive_type_conversion
from python_code.utils.get_sample_from_distribution import get_sample_from_distribution
from python_code.utils.parse_and_repeat import parse_and_repeat

class ClassInstanceSerializer:
    """
    A class dedicated to managing configuration for different class instances.

    This handler is designed for the serialization and deserialization of class instances
    to and from JSON format. It supports saving class instances with their parameters
    to a JSON file and reconstructing the instances from the file. Instances are properly
    typed based on custom mappings and attributes, and initialization parameters are stored
    as ranges, allowing for manual extension or variation for purposes like hyperparameter tuning.

    When the `ClassInstanceSerializer` loads from the JSON file, it randomly selects an element from
    the parameter's range to initialize a class instance. This feature facilitates experimentation
    with different parameter combinations to identify optimal configurations. For example, a parameter
    defined with a range [10] in the JSON can be manually adjusted to [10, 5, 20], from which a value
    will be randomly selected during instantiation.

    Attributes:
        KEY_SEPARATOR (str): A constant separating unique keys in JSON configuration,
                             especially when dealing with multiple instances of the same class.
        instance_mapping (dict): A mapping of class name strings to actual class objects,
                                 used for instantiation from JSON.

    Methods:
        save_instance_list_to_json(instance_list, json_path): Serializes a list of class instances
                                                             into a JSON file.
        get_instance_list_from_json(json_path, additional_init_params=None): Deserializes instances
                                                                             from a JSON file, creating
                                                                             a list of class instances.
        
    The class relies on the presence of a `params` attribute for instances to manage serialization,
    and an optional `arguments_datatype` class attribute for guided deserialization with type conversion.
    An `instance_mapping` dictionary is expected to map from identifiable class names to actual classes.
    """

    KEY_SEPARATOR = '__'       

    def __init__(self, instance_mapping): 
        """
        Initializes a new instance of the ClassInstanceSerializer class.

        This handler manages serialization and deserialization of class instance configurations 
        to and from JSON files. It ensures that class instances are appropriately instantiated
        with their corresponding parameters stored in a JSON format.

        Args:
            instance_mapping (dict): A dictionary mapping class names as strings to the actual 
                                     class objects. This enables the handler to instantiate objects 
                                     of the mapped classes from stored configurations.

        Raises:
            ValueError: If 'instance_mapping' is not of type dict.
        """
        self.instance_mapping = instance_mapping

    @property
    def instance_mapping(self):
        return self._instance_mapping
    
    @instance_mapping.setter
    def instance_mapping(self, value):
        if not type(value) is dict:
            raise ValueError(f"The specified instance mapping is not of type dict: {value}.")
        self._instance_mapping = value

    def save_instance_list_to_json(self, instance_list, json_path):
        """
        Serializes a list of class instances to a JSON file.

        Args:
            instance_list (list): A list of class instances to serialize.
            json_path (str): The file path where the JSON will be saved.
        """
        configs = {}
        for instance in instance_list:
            configs = self._add_instance_to_configs(instance, configs) 
        
        self.save_configs_to_json(configs, json_path)

    def _add_instance_to_configs(self, instance, configs):
        """
        Adds an instance's configuration to the configs dictionary.

        Args:
            instance (object): The class instance to add.
            configs (dict): The dictionary to which the instance's config will be added.

        Returns:
            dict: Updated configurations with the new instance's config added.
        """
        for class_name, mapped_class in self.instance_mapping.items():
            if isinstance(instance, mapped_class):
                class_name = self._generate_unique_key_name(class_name, configs)
                if hasattr(instance, 'params'):
                    configs[class_name] = instance.params
                    return configs
                else:
                    raise AttributeError(f"Mapped class: '{mapped_class}' does not have the attribute 'params'.")
        raise KeyError(f"Instance '{instance}' is not a value in 'instance_mapping' dict.")
    
    def save_configs_to_json(self, configs, json_path): 
        """
        Saves configurations to a JSON file after serializing and formatting.

        Args:
            configs (dict): Dictionary containing class configurations.
            json_path (str): The file path where the JSON will be saved.
        """

        self._verify_json_path(json_path)
        json_data = {}
        for class_name in configs.keys():

            converted_params = {}
            for key, value in configs[class_name].items():
                converted_params[key] = [self._serialize_to_json_value(value)] # Square Brackets required as parameter ranges are expected instead of values.
        
            unique_name = self._generate_unique_key_name(class_name, json_data)   
            json_data[unique_name] = converted_params

        # Write 'json_data' to JSON file in a readable way.
        json_string = json.dumps(json_data, indent=4)
        json_string = json_string.replace('},', '},\n')
        pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'   
        result = re.sub(pattern, self._remove_newlines, json_string)  # Replace newlines and spaces within square brackets.
        with open(json_path, 'w') as file:
            file.write(result)
    
    def _verify_json_path(self, json_path):
        """
        Verifies that the provided path is for a JSON file.

        Args:
            json_path (str): The file path to verify.
        """
        if not json_path.endswith('.json'):
            raise ValueError(f"Specified JSON path '{json_path}' is not JSON.")
         
    def _serialize_to_json_value(self, obj):
        """
        Recursively converts Python objects to JSON serializable types.

        Args:
            obj (object): The Python object to serialize.

        Returns:
            object: The JSON-serializable representation of `obj`.

        """
        if type(obj) is tuple or type(obj) is list:
            return [self._serialize_to_json_value(item) for item in obj]
        elif type(obj) is dict:
            return {key: self._serialize_to_json_value(value) for key, value in obj.items()}
        elif type(obj) in {int, float, str, bool}:
            return obj
        else:
            raise TypeError(f"Object with value '{obj} cannot not be serialized to JSON format.")
        
    def _generate_unique_key_name(self, current_key, dictionary):        
        """
        Generates a unique key name by appending incrementing numbers if a conflict exists.

        Args:
            current_key (str): The base name for the key.
            dictionary (dict): The dictionary which should not have conflicting keys.

        Returns:
            str: A unique key name for the dictionary.
        """
        key = current_key
        i = 2             # Starts from 2 as 1 is the case of key name without identification.
        while key in dictionary.keys():              # Same namining of entries are not allowed in json.
            key = key.split(ClassInstanceSerializer.KEY_SEPARATOR)[0] + ClassInstanceSerializer.KEY_SEPARATOR + str(i) 
            i += 1

        return key
    
    def _remove_newlines(self, match):
        """ 
        Removes newlines and spaces within square brackets in JSON strings.        
        Args:
            match (re.Match): The regex match object containing the matched string.

        Returns:
            str: The matched string with newlines and spaces removed.
        """
        return match.group().replace('\n', '').replace(' ', '')    
        
    def get_instance_list_from_json(self, json_path, additional_init_params=None): 
        """
        Deserializes class instances from a JSON file.

        Args:
            json_path (str): The file path of the JSON to deserialize.
            additional_init_params (dict, optional): Additional initialization parameters to pass to the class constructors.

        Returns:
            list: A list of class instances created from the JSON configuration.
        """

        self._verify_json_path(json_path)
        try:
            with open(json_path, 'r') as file:
                json_data = json.load(file)
                class_names = list(json_data.keys())
        except FileNotFoundError as e:
            raise FileNotFoundError("Specified JSON file storing the instance list to be loaded was not found.") from e
        
        instance_list = []
        for class_name in class_names:
            instance = self._create_class_instance(class_name, json_data, additional_init_params)
            instance_list.append(instance)

        return instance_list
    
    def _create_class_instance(self, class_name, json_data, additional_init_params=None):
        """
        Creates a class instance from the provided JSON data and additional parameters.

        Args:
            class_name (str): The name of the class to instantiate.
            json_data (dict): The JSON data containing the initialization parameters.
            additional_init_params (dict, optional): Additional parameters for the class constructor.

        Returns:
            object: An instance of the class specified by 'class_name'.
        """
        class_name_parts = class_name.split(ClassInstanceSerializer.KEY_SEPARATOR)

        if class_name_parts[0] not in self.instance_mapping.keys():
            raise KeyError(f"Class Name '{class_name_parts[0]}' from json file has no instance mapping.")
        mapped_class = self.instance_mapping[class_name_parts[0]]
        init_params = json_data.get(class_name)
        
        if hasattr(mapped_class, 'arguments_datatype'):

            if type(mapped_class.arguments_datatype) is not dict:
                raise AttributeError(f"The class attribute 'arguments_datatype' of class {mapped_class} must be of type dict. This allows to create a class instance.")
                
            arguments_datatype = mapped_class.arguments_datatype
            init_params = self._deserialize_json_params(init_params)
            init_params = recursive_type_conversion(init_params, arguments_datatype)
        else:
            print(f"Configuration Handler Warning: class '{mapped_class}' has no attribute 'arguments_datatype', this can lead to faulty instanciation.")

        try:
            if additional_init_params:
                return mapped_class(**init_params, **additional_init_params)
            else: 
                return mapped_class(**init_params)
        except ValueError as e:
            raise ValueError(f"Incorrect initialization of class {mapped_class}, probably initialization parameters mismatch with JSON file.") from e
        except TypeError as e:
            raise ValueError(f"Incorrect initialization of class {mapped_class}, probably initialization parameters mismatch with JSON file.") from e

    def _deserialize_json_params(self, json_params):

        deserialized_params = {}
        for param_name, param_val in json_params.items():
            # If param_val is list it indicates a range of parameter values, else if a dict it indicates a distribution.
            if isinstance(param_val, list):
                deserialized_params[param_name] = random.choice(param_val)
            elif isinstance(param_val, dict):
                deserialized_params[param_name] = get_sample_from_distribution(param_val)
            elif isinstance(param_val, str):
                param_range = parse_and_repeat(param_val)
                deserialized_params[param_name] = random.choice(param_range)
            else:
                raise ValueError(f"The value of JSON parameter '{param_name}' must be of type dict or list, not {type(param_val)}.")

        return deserialized_params