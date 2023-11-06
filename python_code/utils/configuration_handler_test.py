import unittest
import os
import shutil

from python_code.utils.configuration_handler import ConfigurationHandler

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
TEST_DIR = os.path.join(ROOT_DIR, r'python_code/utils/test/')


class MockClass1:
    
    _init_params_datatypes = {'param1': str, 'param2': int, 'param3': {'key1': int, 'key2':(float, bool)}}
    def __init__(self, param1, param3, param2=20):
        self.params = {'param1': param1, 'param2': param2, 'param3': param3}
    
    def __eq__(self, obj):
        return self.params == obj.params
    
class MockClass2:
    
    _init_params_datatypes = {'param1': str, 'param2': int}
    def __init__(self, param1, param2=20):
        self.params = {'param1': param1, 'param2': param2}
    
    def __eq__(self, obj):
        return self.params == obj.params

class TestConfigurationHandler(unittest.TestCase):
    """    Test suite for the ConfigurationHandler class.

    This suite contains a set of unit tests that are designed to ensure the proper 
    functionality of the ConfigurationHandler's methods. It tests the ability of 
    the ConfigurationHandler to serialize and deserialize instance configurations, 
    handle different types of inputs, and manage errors and edge cases appropriately.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(TEST_DIR):
            os.mkdir(TEST_DIR)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    def setUp(self):
        self.json_path = os.path.join(TEST_DIR, 'test_config.json')
        with open(self.json_path, 'a'):    pass
        self.instance_mapping = {'MockClass1': MockClass1,'MockClass2': MockClass2}
        self.handler = ConfigurationHandler(self.instance_mapping)
        self.instance_list = [
            MockClass1(param1 = 'hallo', param2 = 20, param3 = {'key1': 30, 'key2':(3.2, True)}),
            MockClass1(param1 = 'tschüss', param3 = {'key1': 40, 'key2':(55.3, False)}), # param2 is expected to be initialized to default value.
            MockClass2(param1 = 'win') # All is expected to be initialized to default value.
            ]
    
    def tearDown(self):
        self.handler.instance_mapping = {'MockClass1': MockClass1,'MockClass2': MockClass2}
        os.remove(self.json_path)
    
    def test_save_instance_list_to_json(self):
        self.handler.save_instance_list_to_json(self.instance_list, self.json_path)
        loaded_instance_list = self.handler.get_instance_list_from_json(self.json_path)
        self.assertEqual(loaded_instance_list, self.instance_list)
    
    def test_mismatch_json_and_class_1(self): 

        with self.assertRaises(ValueError):
            self.handler.save_instance_list_to_json(self.instance_list, self.json_path)
            self.handler.instance_mapping = {'MockClass1': MockClass1,'MockClass2': MockClass1}   # Purposly wrong mapping for init params mismatch.
            loaded_instance_list = self.handler.get_instance_list_from_json(self.json_path)
            self.assertEqual(loaded_instance_list, self.instance_list)
   
    def test_mismatch_json_and_class_2(self): 

        self.handler.instance_mapping = {'MockClass1': MockClass1}   # Missing mapping for 'MockClass2'.
        with self.assertRaises(KeyError):
            self.handler.save_instance_list_to_json(self.instance_list, self.json_path)

    def test_serialize_success_1(self):
        self.assertEqual(self.handler._serialize_to_json_value([1, 2, 3]), [1, 2, 3])
        self.assertEqual(self.handler._serialize_to_json_value((1, 2, 3)), [1, 2, 3])
        self.assertEqual(self.handler._serialize_to_json_value({'a': 1, 'b': 2}), {'a': 1, 'b': 2})
        self.assertEqual(self.handler._serialize_to_json_value(1), 1)
        self.assertEqual(self.handler._serialize_to_json_value(1.0), 1.0)
        self.assertEqual(self.handler._serialize_to_json_value("test"), "test")
        self.assertEqual(self.handler._serialize_to_json_value(True), True)

    def test_serialize_success_2(self):
        nested_structure = {
            'list': [1, 2, 3],
            'tuple': (1, 2, 3),
            'dict': {'nested_list': [4, 5, 6]}
        }
        expected = {
            'list': [1, 2, 3],
            'tuple': [1, 2, 3],
            'dict': {'nested_list': [4, 5, 6]}
        }
        self.assertEqual(self.handler._serialize_to_json_value(nested_structure), expected)

    def test_serialize_failed(self):
        with self.assertRaises(TypeError):
            self.handler._serialize_to_json_value(set([1, 2, 3]))
        class CustomObject:
            pass
        with self.assertRaises(TypeError):
            self.handler._serialize_to_json_value(CustomObject())


        

if __name__ == '__main__':
    unittest.main()
