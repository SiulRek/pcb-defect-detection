import tensorflow as tf


class LabelManager:
    """
    Manages different types of label encoding for machine learning models.

    Attributes:
        label_type (str): The type of labels managed, which determines the encoding method used.
        num_classes (int, optional): The number of classes used for categorical label encoding.
    
    Methods:
        get_label: Depending on the `label_type`, it delegates to the corresponding method to fetch 
            labels.
    """

    def __init__(self, label_type, num_classes=None):
        """
        Initializes the LabelManager with a specific label encoding type and, optionally, the number
        of classes.

        Args:
            label_type (str): The type of label encoding to manage. Supported types are 
                'category_codes','sparse_category_codes', and 'object_detection'.
            num_classes (int, optional): The total number of classes, necessary for categorical 
                encodings.Required if label_type is 'category_codes'.
        """
        self.label_type = label_type
        self.num_classes = num_classes
        if label_type == 'category_codes':
            self.get_label = self.get_categorical_labels
        elif label_type == 'sparse_category_codes':
            self.get_label = self.get_sparse_categorical_labels
        elif label_type == 'object_detection':
            self.get_label = self.get_object_detection_labels
    
    def get_categorical_labels(self, sample):
        """
        Encodes categorical labels into one-hot vectors.

        Args:
            sample (dict): A dictionary containing the label data with key 'category_codes'.

        Returns:
            tf.Tensor: A one-hot encoded TensorFlow constant of the label.
        """
        try:
            label = int(sample['category_codes'])
            label = tf.constant(label, dtype=tf.int8)
            label = tf.one_hot(label, self.num_classes)
            return label 
        except KeyError as e:
            msg = "The sample dictionary does not contain the key 'category_codes'."
            raise KeyError(msg) from e
        except ValueError as e:
            msg = "The 'category_codes' key in sample should be convertable to integer."
            raise ValueError(msg) from e
        # Except num_classes is invalid
        except tf.errors.OpError as e:
            msg = "Probably the number of classes is invalid."
            raise ValueError(msg) from e
        
    def get_sparse_categorical_labels(self, sample):
        """
        Encodes categorical labels into sparse format suitable for sparse categorical crossentropy.

        Args:
            sample (dict): A dictionary containing the label data with key 'category_codes'.

        Returns:
            tf.Tensor: A TensorFlow constant of the label in sparse format.
        """
        try:
            label = int(sample['category_codes'])
            label = tf.constant(label, dtype=tf.int8)
            return label
        except KeyError as e:
            msg = "The sample dictionary does not contain the key 'category_codes'."
            raise KeyError(msg) from e
        except ValueError as e:
            msg = "The 'category_codes' key in sample should be convertable to integer."
            raise ValueError(msg) from e
        
    def get_object_detection_labels(self, sample):
        """
        Stub method for future implementation of object detection label encoding.

        Args:
            sample (dict): A dictionary containing the label data.

        Raises:
            NotImplementedError: Indicates that the method is not yet implemented.
        """
        raise NotImplementedError("Object Detection Labels are not yet implemented.")
