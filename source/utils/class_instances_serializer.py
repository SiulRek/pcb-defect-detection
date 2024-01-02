import os
import re
import json
import random

from source.utils.recursive_type_conversion import recursive_type_conversion
from source.utils.get_sample_from_distribution import get_sample_from_distribution
from source.utils.parse_and_repeat import parse_and_repeat
from source.utils.randomly_select_sequential_keys import randomly_select_sequential_keys

class ClassInstancesSerializer:
    """
    Manages serialization and deserialization of class instances to and from JSON format. 
    Supports saving class instances with parameters to JSON and reconstructing them with 
    optional parameter randomization. Useful for hyperparameter tuning and experimentation.

    Attributes:
        KEY_SEPARATOR (str): Separator for constructing unique keys in JSON files.
        instance_mapping (dict): Maps class names to actual class objects.

    Public Methods:
        save_instances_to_json(instance_list, json_path): Serializes class instances to JSON.
        get_instances_from_json(json_path): Deserializes instances from JSON with specific parameters.
        get_randomized_instances_from_json(json_path): Deserializes instances from JSON with randomized parameters.

    Note: 
    This class relies on 'parameters' and optionally 'arguments_datatype' in class instances for serialization and deserialization.
    The 'instance_mapping' is essential for class instantiation.
    """

    KEY_SEPARATOR = '__'       

    def __init__(self, instance_mapping): 
        """
        Initializes a new instance of the ClassInstancesSerializer class.

        This handler manages serialization and deserialization of class instance configurations 
        to and from JSON files. It ensures that class instances are appropriately instantiated
        with their corresponding parameters stored in a JSON format.

        Args:
            instance_mapping (dict): A dictionary mapping class names as strings to the actual 
                                     class objects. This enables the handler to instantiate objects 
                                     of the mapped classes from stored configurations.
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

    def _verify_json_path(self, json_path):
        """
        Verifies that the provided path is a JSON file and the base directory of the file exists.

        Args:
            json_path (str): The file path to verify.
        """
        if not os.path.exists(os.path.dirname(json_path)):
            raise ValueError(f"The Base Directory of JSON path {json_path} does not exists.")
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

    def _add_instance_to_configurations(self, instance, configurations):
        """
        Adds an instance's configuration to the configurations dictionary.

        Args:
            instance (object): The class instance to add.
            configurations (dict): The dictionary to which the instance's config will be added.

        Returns:
            dict: Updated class instance configurations with the new instance's config added.
        """
        for class_name, mapped_class in self.instance_mapping.items():
            if isinstance(instance, mapped_class):
                class_name = self._generate_unique_key_name(class_name, configurations)
                if not hasattr(instance, 'parameters'):
                    raise AttributeError(f"Mapped class: '{mapped_class}' does not have the attribute 'parameters'.")
                if not isinstance(instance.parameters, dict):
                    raise AttributeError(f"Mapped class: '{mapped_class}' attribute 'parameters' is not of type dict.")
                configurations[class_name] = instance.parameters
                return configurations
        raise KeyError(f"Instance '{instance}' is not a value in 'instance_mapping' dict.")
    
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
            key = key.split(ClassInstancesSerializer.KEY_SEPARATOR)[0] + ClassInstancesSerializer.KEY_SEPARATOR + str(i) 
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
    
    def _save_configurations_to_json(self, configurations, json_path): 
        """
        Saves configurations of the class instances to a JSON file after serializing and formatting.

        Args:
            configurations (dict): Dictionary containing the class instance configuration.
            json_path (str): The file path where the JSON will be saved.
        """

        self._verify_json_path(json_path)
        json_data = {}
        for class_name in configurations.keys():

            converted_parameters = {}
            for key, value in configurations[class_name].items():
                converted_parameters[key] = self._serialize_to_json_value(value) # Square Brackets required as parameter ranges are expected instead of values.
        
            unique_name = self._generate_unique_key_name(class_name, json_data)   
            json_data[unique_name] = converted_parameters

        # Write 'json_data' to JSON file in a readable way.
        json_string = json.dumps(json_data, indent=4)
        json_string = json_string.replace('},', '},\n')
        pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'   
        result = re.sub(pattern, self._remove_newlines, json_string)  # Replace newlines and spaces within square brackets.
        with open(json_path, 'w') as file:
            file.write(result)
    
    def save_instances_to_json(self, instance_list, json_path):
        """
        Serializes a list of class instances to a JSON file.

        Args:
            instance_list (list): A list of class instances to serialize.
            json_path (str): The file path where the JSON will be saved.
        """
        configurations = {}
        for instance in instance_list:
            configurations = self._add_instance_to_configurations(instance, configurations) 
        
        self._save_configurations_to_json(configurations, json_path)

    def _deserialize_json_parameters(self, json_parameters, randomized):

        deserialized_parameters = {}
        for param_name, param_val in json_parameters.items():
            # If param_val is list it indicates a range of parameter values, else if a dict it indicates a distribution.
            if not randomized:
                deserialized_parameters[param_name] = param_val
            elif isinstance(param_val, list):
                deserialized_parameters[param_name] = random.choice(param_val)
            elif isinstance(param_val, dict):
                deserialized_parameters[param_name] = get_sample_from_distribution(param_val)
            elif isinstance(param_val, str):
                param_range = parse_and_repeat(param_val)
                deserialized_parameters[param_name] = random.choice(param_range)
            else:
                raise ValueError(f"The value of JSON parameter '{param_name}' must be of type dict, list or str not {type(param_val)}.")

        return deserialized_parameters

    def _extract_arguments(self, json_data, class_name, mapped_class, randomized):
        """    Extracts and converts initialization parameters for a class instance from JSON data.

        Args:
            json_data (dict): The JSON data from which to extract initialization parameters.
            class_name (str): The name of the class for which parameters are being extracted.
            mapped_class (type): The class object associated with 'class_name'.
            randomized (bool): Determines whether parameters should be randomized from range.
        
        Returns:
            dict: A dictionary of deserialized and type-converted initialization parameters for the class.
        """

        json_parameters = json_data.get(class_name)
        arguments = self._deserialize_json_parameters(json_parameters, randomized)

        if hasattr(mapped_class, 'arguments_datatype'):

            if type(mapped_class.arguments_datatype) is not dict:
                raise AttributeError(f"The class attribute 'arguments_datatype' of class {mapped_class} must be of type dict. This allows to create a class instance.")
                
            arguments_datatype = mapped_class.arguments_datatype
            arguments = recursive_type_conversion(arguments, arguments_datatype)
        else:
            print(f"Class Instance Serializer Warning: class '{mapped_class}' has no attribute 'arguments_datatype', this can lead to faulty instanciation.")
        
        return arguments

    def _create_class_instance(self, class_name, json_data, randomized):
        """
        Creates a class instance from the provided JSON data and additional parameters.

        Args:
            class_name (str): The name of the class to instantiate.
            json_data (dict): The JSON data containing the initialization parameters.
            randomized (bool): Determines whether parameters should be randomized from range.

        Returns:
            object: An instance of the class specified by 'class_name'.
        """

        class_name_parts = class_name.split(ClassInstancesSerializer.KEY_SEPARATOR)

        if class_name_parts[0] not in self.instance_mapping.keys():
            raise KeyError(f"Class Name '{class_name_parts[0]}' from json file has no instance mapping.")
        mapped_class = self.instance_mapping[class_name_parts[0]]
        
        arguments = self._extract_arguments(json_data, class_name, mapped_class, randomized)

        try:
            return mapped_class(**arguments)
        except ValueError as e:
            raise ValueError(f"Incorrect instanciation of class {mapped_class}, probably due to arguments mismatch with JSON file or incorrect mapping.") from e
        except TypeError as e:
            raise ValueError(f"Incorrect instanciation of class {mapped_class}, probably due to arguments mismatch with JSON file or incorrect mapping.") from e

    def _build_instances_from_json(self, json_path, randomized):
        """
        Builds class instances from a JSON file. Depending on the 'randomized'
        flag, it either uses specific parameters or randomly selects parameters from specified ranges.

        Args:
            json_path (str): The file path of the JSON to deserialize.
            randomized (bool): Determines whether parameters should be randomized.

        Returns:
            list: A list of class instances created from the JSON file.
        """
        self._verify_json_path(json_path)
        try:
            with open(json_path, 'r') as file:
                json_data = json.load(file)
                json_data = randomly_select_sequential_keys(json_data, separator=ClassInstancesSerializer.KEY_SEPARATOR)
                class_names = list(json_data.keys())
        except FileNotFoundError as e:
            raise FileNotFoundError("Specified JSON file storing the instance list to be loaded was not found.") from e
        except KeyError as e:
            raise KeyError("The JSON file storing the instance list to be loaded has faulty key names.") from e
        
        instance_list = []
        for class_name in class_names:
            instance = self._create_class_instance(class_name, json_data, randomized=randomized)
            instance_list.append(instance)

        return instance_list

    def get_instances_from_json(self, json_path):
        """
        Deserializes class instances from a JSON file with specific parameters.

        Args:
            json_path (str): The file path of the JSON to deserialize.

        Returns:
            list: A list of class instances created from the JSON file with specific parameters.
        """
        return self._build_instances_from_json(json_path, randomized=False)

    def get_randomized_instances_from_json(self, json_path):
        """
        Deserializes class instances from a JSON file with randomly selected parameters.

        Args:
            json_path (str): The file path of the JSON to deserialize.

        Returns:
            list: A list of class instances created from the JSON file with random parameters.
        """ 
        return self._build_instances_from_json(json_path, randomized=True)

