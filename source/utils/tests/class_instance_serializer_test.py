import unittest
import os
import json

from source.utils.class_instances_serializer import ClassInstancesSerializer
from source.utils.test_result_logger import TestResultLogger


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','..')
OUTPUT_DIR = os.path.join(ROOT_DIR, r'source/utils/tests/outputs')
LOG_FILE = os.path.join(OUTPUT_DIR, 'test_results.log')


class MockClass1:
    
    arguments_datatype = {'param1': str, 'param2': int, 'param3': {'key1': int, 'key2':(float, bool)}}
    def __init__(self, param1, param3, param2=20):
        self.parameters = {'param1': param1, 'param2': param2, 'param3': param3}
    
    def __eq__(self, obj):
        return self.parameters == obj.parameters
    
    
class MockClass2:
    
    arguments_datatype = {'param1': str, 'param2': int}
    def __init__(self, param1, param2=20):
        self.parameters = {'param1': param1, 'param2': param2}
        self.name = 'MockClass2'
    
    def __eq__(self, obj):
        return self.parameters == obj.parameters


class MockClassWithoutArgsSpec:
    
    def __init__(self, param1 = 3):
        self.parameters = {'param1': param1}

    def __eq__(self, obj):
        return self.parameters == obj.parameters


class MockClassWithoutparametersAttr: pass


class MockClassInvalidparametersAttr:

    def __init__(self, param1 = 3):
        self.parameters = param1


class TestClassInstancesSerializer(unittest.TestCase):
    """    Test suite for the ClassInstancesSerializer class.

    This suite contains a set of unit tests that are designed to ensure the proper 
    functionality of the ClassInstancesSerializer's methods. It tests the ability of 
    the ClassInstancesSerializer to serialize and deserialize instance configurations, 
    handle different types of inputs, and manage errors and edge cases appropriately.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = TestResultLogger(LOG_FILE, 'class Instance Serializer Test')

    def setUp(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.json_path = os.path.join(OUTPUT_DIR, 'test_config.json')
        with open(self.json_path, 'a'):    pass
        self.instance_mapping = {'MockClass1': MockClass1,'MockClass2': MockClass2}
        self.serializer = ClassInstancesSerializer(self.instance_mapping)
        self.instance_list = [
            MockClass1(param1 = 'hallo', param2 = 20, param3 = {'key1': 30, 'key2':(3.2, True)}),
            MockClass1(param1 = 'tschuess', param3 = {'key1': 40, 'key2':(55.3, False)}), # param2 is expected to be initialized to default value.
            MockClass2(param1 = 'win') # All is expected to be initialized to default value.
            ]
    
    def tearDown(self):
        os.remove(self.json_path)
        self.logger.log_test_outcome(self._outcome.result, self._testMethodName)

    def test_serialize_success_1(self):
        """ 
        Test the serialization to JSON value of a variety of different data types.
        """
        self.assertEqual(self.serializer._serialize_to_json_value([1, 2, 3]), [1, 2, 3])
        self.assertEqual(self.serializer._serialize_to_json_value((1, 2, 3)), [1, 2, 3])
        self.assertEqual(self.serializer._serialize_to_json_value({'a': 1, 'b': 2}), {'a': 1, 'b': 2})
        self.assertEqual(self.serializer._serialize_to_json_value(1), 1)
        self.assertEqual(self.serializer._serialize_to_json_value(1.0), 1.0)
        self.assertEqual(self.serializer._serialize_to_json_value("test"), "test")
        self.assertEqual(self.serializer._serialize_to_json_value(True), True)

    def test_serialize_success_2(self):
        """ 
        Test the serialization to JSON value of a variety of a nested structure.
        """
        nested_structure = {
            'list': [1, 2, 3],
            'tuple': (1, 2, 3),
            'dict': {'nested_list': (4, 5, '30')}
        }
        expected = {
            'list': [1, 2, 3],
            'tuple': [1, 2, 3],
            'dict': {'nested_list': [4, 5, '30']}
        }
        self.assertEqual(self.serializer._serialize_to_json_value(nested_structure), expected)

    def test_serialize_failed(self):
        """ 
        Test the serialization to JSON value of a variety of invalid data types leading to failure.
        """
        with self.assertRaises(TypeError):
            self.serializer._serialize_to_json_value(set([1, 2, 3]))
        class CustomObject:
            pass
        with self.assertRaises(TypeError):
            self.serializer._serialize_to_json_value(CustomObject())

    def test_deserialize_json_parameters(self):
        """ 
        Test the deserialization of JSON parameters without randomized values.
        """
        source = {
            "number_str": "123",
            "list_of_int": [1, 2, 3],
            "nested_dict": {
                "bool_str": True
            },
            "tuple_of_mixed": ('30','',['30',10])
        }
        expected = {
            "number_str": "123",
            "list_of_int": [1, 2, 3],
            "nested_dict": {
                "bool_str": True
            },
            "tuple_of_mixed": ('30','',['30',10])
        }

        output = self.serializer._deserialize_json_parameters(source, randomized=False)
        self.assertEqual(output, expected)

    def test_deserialize_json_parameters_randomized_1(self):
        """ 
        Test the deserialization of JSON parameters with randomized values.
        """
        source = {
            "number_str": ["123"],
            "list_of_int": [[1, 2, 3]],
            "nested_dict": [{
                "bool_str": True
            }],
            "tuple_of_mixed": [('30','',['30',10])]
        }
        expected = {
            "number_str": "123",
            "list_of_int": [1, 2, 3],
            "nested_dict": {
                "bool_str": True
            },
            "tuple_of_mixed": ('30','',['30',10])
        }

        output = self.serializer._deserialize_json_parameters(source, randomized=True)
        self.assertEqual(output, expected)
        
    def test_deserialize_json_parameters_randomized_2(self):
        """ 
        Test the deserialization of JSON parameters with randomized values.
        """
        source = {
            "num_1": {'distribution': 'uniform', 'low': 2, 'high': 10},
            "num_2": {'distribution': 'uniform', 'low': 0, 'high': 5},
            "num_3" : {'distribution': 'uniform', 'low': -10, 'high': 2}
        }

        output = self.serializer._deserialize_json_parameters(source, randomized=True)
        self.assertTrue(2 <= output['num_1'] <= 10)
        self.assertTrue(0 <= output['num_2'] <= 5)
        self.assertTrue(-10 <= output['num_3'] <= 2)

    def test_deserialize_json_parameters_randomized_3(self):
        """ 
        Test the deserialization of JSON parameters with randomized values.
        """
        source = {
            "param_1": '[8]*2 + [10]*1',
            "param_2": '[1]*3 + [4]*2 + [True] + ["String"]',
            "param_3" : '[""] + ["World", "True"]*2 + [["World", "True"]]*2'
        }
        expected = {
            "param_1": [8,8,10],
            "param_2": [1, 1, 1, 4, 4, True, "String"],
            "param_3" : ['',"World", "True","World", "True",["World", "True"], ["World", "True"]]
        }

        output = self.serializer._deserialize_json_parameters(source, randomized=True)
        self.assertIn(output['param_1'], expected['param_1'])
        self.assertIn(output['param_2'], expected['param_2'])
        self.assertIn(output['param_3'], expected['param_3'])

    def test_creation_of_json(self):
        """ 
        Test the creation of a JSON file.
        """
        json = os.path.join(OUTPUT_DIR, 'serializer_test_file.json')
        self.serializer.save_instances_to_json([],json) # Now a File is created.
        os.remove(json)
    
    def test_load_from_json(self):
        """ 
        Test the loading of instances from a JSON file.
        """
        mock_class_parameters_1 = {'param1': 'hallo', 'param2': 20}     
        mock_class_parameters_2 = {'param1': 'servus', 'param2': 3}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_parameters_1, temp_key:mock_class_parameters_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)
        loaded_instance_list = self.serializer.get_instances_from_json(self.json_path)
        
        self.assertEqual(loaded_instance_list[0].parameters['param1'], mock_class_parameters_1['param1'])
        self.assertEqual(loaded_instance_list[0].parameters['param2'], mock_class_parameters_1['param2'])
        self.assertEqual(loaded_instance_list[1].parameters['param1'], mock_class_parameters_2['param1'])
        self.assertTrue(isinstance(loaded_instance_list[0].parameters['param2'], int))
        self.assertTrue(isinstance(loaded_instance_list[1].parameters['param2'], int))
  
    def test_load_randomized_json(self):
        """ 
        Test the loading of instances from a JSON file with randomized values.
        """
        mock_class_parameters_1 = {'param1': ['tschuess','hallo'], 'param2': [20,30,40]}     
        mock_class_parameters_2 = {'param1': ['servus', 'ciao'], 'param2': {'distribution': 'uniform', 'low': 1, 'high':10}}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_parameters_1, temp_key:mock_class_parameters_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)

        loaded_instance_list = self.serializer.get_randomized_instances_from_json(self.json_path)
        
        self.assertIn(loaded_instance_list[0].parameters['param1'], mock_class_parameters_1['param1'])
        self.assertIn(loaded_instance_list[0].parameters['param2'], mock_class_parameters_1['param2'])
        self.assertIn(loaded_instance_list[1].parameters['param1'], mock_class_parameters_2['param1'])
        self.assertTrue(1 <= loaded_instance_list[1].parameters['param2'] <= 10)

    def test_key_randomization_with_json(self):
        """ 
        Test the randomization of keys in a JSON file.
        """
        mock_class_parameters_1 = {'param1': 'hallo', 'param2': 20, 'param3': {'key1': 30, 'key2':(3.2, True)}}     
        mock_class_parameters_2 = {'param1': 'servus', 'param2': 3} 
        sep = ClassInstancesSerializer.KEY_SEPARATOR  
        key_1 = 'MockClass1' + sep + 'i0' + sep + 'I0'               
        key_2 = 'MockClass2' + sep + 'i0' + sep + 'I0F9'               
        key_3 = 'MockClass1' + sep + 'i1' + sep + 'I1'               
        key_4 = 'MockClass2' + sep + 'i1' + sep + 'I1F9'               
        json_data = {
            key_1: mock_class_parameters_1, 
            key_2: mock_class_parameters_2,
            key_3: mock_class_parameters_1,
            key_4: mock_class_parameters_2
            }
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)

        loaded_instance_lists = []
        for _ in range(500):
            loaded_instance_lists.append(self.serializer.get_instances_from_json(self.json_path))
        
        key_counts = {'MockClass1': 0, 'MockClass2': 0}
        for lst in loaded_instance_lists:
            self.assertEqual(len(lst), 2)
            for instance in lst:
                if isinstance(instance, MockClass1):
                    key_counts['MockClass1'] += 1
                if isinstance(instance, MockClass2):
                    key_counts['MockClass2'] += 1

        self.assertAlmostEqual(key_counts['MockClass1'], 100, delta=25)
        self.assertAlmostEqual(key_counts['MockClass2'], 900, delta=25)
    
    def test_save_instances_to_json(self):
        """ 
        Test the saving of instances to a JSON file.
        """
        self.serializer.save_instances_to_json(self.instance_list, self.json_path)
        loaded_instance_list = self.serializer.get_instances_from_json(self.json_path)
        self.assertEqual(loaded_instance_list, self.instance_list)

    def test_invalid_json_path(self):
        """ 
        Test the handling of invalid JSON paths.
        """
        invalid_paths = [
            ('directory/does/not/exists/test_config.json', ValueError),
            ('source/utils/test_config.txt', ValueError),
        ]

        for path, expected_exception in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaises(expected_exception):
                    self.serializer._verify_json_path(path)
                with self.assertRaises(expected_exception):
                    self.serializer.get_instances_from_json(path)
                with self.assertRaises(expected_exception):
                    self.serializer.get_randomized_instances_from_json(path)
                with self.assertRaises(expected_exception):
                    self.serializer.save_instances_to_json([], path)
        
        path_dir_exists_file_not = 'source/utils/not_existing_file.json'
        with self.assertRaises(FileNotFoundError):
            self.serializer.get_instances_from_json(path_dir_exists_file_not)
        with self.assertRaises(FileNotFoundError):
            self.serializer.get_randomized_instances_from_json(path_dir_exists_file_not)
    
    def test_instanciation_with_default_from_json(self):
        """ 
        Test the instanciation of instances from a JSON file with default values.
        """
        mock_class_parameters_1 = {'param1': 'tschuess'}     # 'param2' is not specified in JSON, initialization to default is expected.
        mock_class_parameters_2 = {'param1': 'servus'}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_parameters_1, temp_key:mock_class_parameters_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)
        loaded_instance_list = self.serializer._build_instances_from_json(self.json_path, randomized=False)
        
        self.assertEqual(loaded_instance_list[0].parameters['param1'], mock_class_parameters_1['param1'])
        self.assertEqual(loaded_instance_list[1].parameters['param1'], mock_class_parameters_2['param1'])
        self.assertTrue(isinstance(loaded_instance_list[1].parameters['param2'], int))
   
    def test_missing_mapping_in_save(self): 
        """ 
        Test the saving of instances to a JSON file with a missing mapping leading to Error.
        """
        with self.assertRaises(KeyError):
            self.serializer.instance_mapping = {'MockClass1': MockClass1}   # Missing mapping for 'MockClass2'.
            self.serializer.save_instances_to_json(self.instance_list, self.json_path) 

    def test_missing_mapping_in_load(self): 
        """ 
        Test the loading of instances from a JSON file with a missing mapping leading to Error.
        """
        mock_class_parameters = {'param1': ['tschuess','hallo'], 'param2': [20,30,40]}     
        json_data = {'MockClass2': mock_class_parameters}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)

        self.serializer.instance_mapping = {'MockClass1': MockClass1}   # Missing mapping for 'MockClass2'.
        with self.assertRaises(KeyError):
            self.serializer.get_instances_from_json(self.json_path) 
        with self.assertRaises(KeyError):
            self.serializer.get_randomized_instances_from_json(self.json_path) 

    def test_invalid_parameter_range_in_json(self):
        """ 
        Test the handling of invalid parameter ranges in a JSON file leading to error.
        """
        mock_class_parameters_1 = {'param1': ['tschuess'], 'param2': [20], 'invalid_param': 10}     
        mock_class_parameters_2 = {'param1': ['servus'], 'param2': [30]}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_parameters_1, temp_key:mock_class_parameters_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)
        with self.assertRaises(ValueError):
            self.serializer.get_randomized_instances_from_json(self.json_path)

    def test_no_argument_specification(self):
        """
        Test the serialization of instances without argument datatype specification.
        """
        instance_list = [MockClassWithoutArgsSpec(param1=1)]
        self.serializer.instance_mapping = {'MockClassWithoutArgsSpec': MockClassWithoutArgsSpec}   
        self.serializer.save_instances_to_json(instance_list, self.json_path)
        loaded_instance_list = self.serializer._build_instances_from_json(self.json_path, randomized=False)
        self.assertEqual(loaded_instance_list, instance_list)       

    def test_missing_attribute_parameters(self):
        """ 
        Test the serialization of instances with missing attribute parameters leading to error. 
        """
        instance_list = [MockClassWithoutparametersAttr()]
        self.serializer.instance_mapping = {'MockClassWithoutparametersAttr': MockClassWithoutparametersAttr}   
        with self.assertRaises(AttributeError):
            self.serializer.save_instances_to_json(instance_list, self.json_path)
    
    def test_invalid_attribute_parameters(self): 
        """ 
        Test the serialization of instances with invalid attribute parameters leading to error.
        """
        instance_list = [MockClassInvalidparametersAttr(param1=1)]
        self.serializer.instance_mapping = {'MockClassInvalidparametersAttr': MockClassInvalidparametersAttr}   
        with self.assertRaises(AttributeError):
            self.serializer.save_instances_to_json(instance_list, self.json_path)

    
if __name__ == '__main__':
    unittest.main()
