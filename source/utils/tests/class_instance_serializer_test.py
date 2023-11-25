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
        self.params = {'param1': param1, 'param2': param2, 'param3': param3}
    
    def __eq__(self, obj):
        return self.params == obj.params
    
    
class MockClass2:
    
    arguments_datatype = {'param1': str, 'param2': int}
    def __init__(self, param1, param2=20):
        self.params = {'param1': param1, 'param2': param2}
    
    def __eq__(self, obj):
        return self.params == obj.params


class MockClassWithoutArgsSpec:
    
    def __init__(self, param1 = 3):
        self.params = {'param1': param1}

    def __eq__(self, obj):
        return self.params == obj.params


class MockClassWithoutParamsAttr: pass


class MockClassInvalidParamsAttr:

    def __init__(self, param1 = 3):
        self.params = param1


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
        self.assertEqual(self.serializer._serialize_to_json_value([1, 2, 3]), [1, 2, 3])
        self.assertEqual(self.serializer._serialize_to_json_value((1, 2, 3)), [1, 2, 3])
        self.assertEqual(self.serializer._serialize_to_json_value({'a': 1, 'b': 2}), {'a': 1, 'b': 2})
        self.assertEqual(self.serializer._serialize_to_json_value(1), 1)
        self.assertEqual(self.serializer._serialize_to_json_value(1.0), 1.0)
        self.assertEqual(self.serializer._serialize_to_json_value("test"), "test")
        self.assertEqual(self.serializer._serialize_to_json_value(True), True)

    def test_serialize_success_2(self):
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
        with self.assertRaises(TypeError):
            self.serializer._serialize_to_json_value(set([1, 2, 3]))
        class CustomObject:
            pass
        with self.assertRaises(TypeError):
            self.serializer._serialize_to_json_value(CustomObject())

    def test_deserialize_json_params_1(self):
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

        output = self.serializer._deserialize_json_params(source)
        self.assertEqual(output, expected)
        
    def test_deserialize_json_params_2(self):
        source = {
            "num_1": {'distribution': 'uniform', 'low': 2, 'high': 10},
            "num_2": {'distribution': 'uniform', 'low': 0, 'high': 5},
            "num_3" : {'distribution': 'uniform', 'low': -10, 'high': 2}
        }

        output = self.serializer._deserialize_json_params(source)
        self.assertTrue(2 <= output['num_1'] <= 10)
        self.assertTrue(0 <= output['num_2'] <= 5)
        self.assertTrue(-10 <= output['num_3'] <= 2)

    def test_deserialize_json_params_3(self):
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

        output = self.serializer._deserialize_json_params(source)
        self.assertIn(output['param_1'], expected['param_1'])
        self.assertIn(output['param_2'], expected['param_2'])
        self.assertIn(output['param_3'], expected['param_3'])

        
    def test_invalid_json_path(self):
        invalid_paths = [
            ('directory/does/not/exists/test_config.json', ValueError),
            ('source/utils/test_config.txt', ValueError),
        ]

        for path, expected_exception in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaises(expected_exception):
                    self.serializer._verify_json_path(path)
                with self.assertRaises(expected_exception):
                    self.serializer.generate_instance_list_from_json(path)
                with self.assertRaises(expected_exception):
                    self.serializer.save_instance_list_to_json([], path)
        
        with self.assertRaises(FileNotFoundError):
            path_dir_exists_file_not = 'source/utils/not_existing_file.json'
            self.serializer.generate_instance_list_from_json(path_dir_exists_file_not)
        self.serializer.save_instance_list_to_json([],path_dir_exists_file_not) # Now File is created.
        os.remove(path_dir_exists_file_not)
    
    def test_load_from_json(self):
        
        mock_class_params_1 = {'param1': ['tschuess','hallo'], 'param2': [20,30,40]}     
        mock_class_params_2 = {'param1': ['servus', 'ciao'], 'param2': {'distribution': 'uniform', 'low': 1, 'high':10}}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_params_1, temp_key:mock_class_params_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)
        loaded_instance_list = self.serializer.generate_instance_list_from_json(self.json_path)
        
        self.assertIn(loaded_instance_list[0].params['param1'], mock_class_params_1['param1'])
        self.assertIn(loaded_instance_list[0].params['param2'], mock_class_params_1['param2'])
        self.assertIn(loaded_instance_list[1].params['param1'], mock_class_params_2['param1'])
        self.assertTrue(isinstance(loaded_instance_list[1].params['param2'], int))
        self.assertTrue(1 <= loaded_instance_list[1].params['param2'] <= 10)

    def test_save_instance_list_to_json(self):
        self.serializer.save_instance_list_to_json(self.instance_list, self.json_path)
        loaded_instance_list = self.serializer.generate_instance_list_from_json(self.json_path)
        self.assertEqual(loaded_instance_list, self.instance_list)
   
    def test_missing_mapping(self): 
        with self.assertRaises(KeyError):
            self.serializer.instance_mapping = {'MockClass1': MockClass1}   # Missing mapping for 'MockClass2'.
            self.serializer.save_instance_list_to_json(self.instance_list, self.json_path) 
    
    def test_instanciation_with_default_from_json(self):
        mock_class_params_1 = {'param1': ['tschuess']}     # 'param2' is not specified in JSON, initialization to default is expected.
        mock_class_params_2 = {'param1': ['servus']}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_params_1, temp_key:mock_class_params_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)
        loaded_instance_list = self.serializer.generate_instance_list_from_json(self.json_path)
        
        self.assertIn(loaded_instance_list[0].params['param1'], mock_class_params_1['param1'])
        self.assertIn(loaded_instance_list[1].params['param1'], mock_class_params_2['param1'])
        self.assertTrue(isinstance(loaded_instance_list[1].params['param2'], int))

    def test_invalid_parameter_in_json(self):
        mock_class_params_1 = {'param1': ['tschuess'], 'param2': [20], 'invalid_param': 10}     
        mock_class_params_2 = {'param1': ['servus'], 'param2': [30]}   
        temp_key = 'MockClass2' + ClassInstancesSerializer.KEY_SEPARATOR + '2'              # As two keys of the same name are not allowed.
        json_data = {'MockClass2': mock_class_params_1, temp_key:mock_class_params_2}
        
        with open(self.json_path, 'w') as file:
            json.dump(json_data, file)
        with self.assertRaises(ValueError):
            self.serializer.generate_instance_list_from_json(self.json_path)

    def test_no_argument_specification(self):
        instance_list = [MockClassWithoutArgsSpec(param1=1)]
        self.serializer.instance_mapping = {'MockClassWithoutArgsSpec': MockClassWithoutArgsSpec}   
        self.serializer.save_instance_list_to_json(instance_list, self.json_path)
        loaded_instance_list = self.serializer.generate_instance_list_from_json(self.json_path)
        self.assertEqual(loaded_instance_list, instance_list)       

    def test_missing_attribute_params(self):
        instance_list = [MockClassWithoutParamsAttr()]
        self.serializer.instance_mapping = {'MockClassWithoutParamsAttr': MockClassWithoutParamsAttr}   
        with self.assertRaises(AttributeError):
            self.serializer.save_instance_list_to_json(instance_list, self.json_path)
    
    def test_invalid_attribute_params(self): 
        instance_list = [MockClassInvalidParamsAttr(param1=1)]
        self.serializer.instance_mapping = {'MockClassInvalidParamsAttr': MockClassInvalidParamsAttr}   
        with self.assertRaises(AttributeError):
            self.serializer.save_instance_list_to_json(instance_list, self.json_path)

    
        
        

if __name__ == '__main__':
    unittest.main()
