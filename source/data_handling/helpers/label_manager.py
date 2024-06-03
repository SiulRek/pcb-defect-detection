import tensorflow as tf


class LabelManager:
    """
    Manages different types of label encoding for machine learning models.

    Attributes:
        - label_type (str): The type of labels managed, which determines the
            encoding method used.
        - num_classes (int, optional): The number of classes used for
            categorical label encoding.

    Methods:
        - get_label: Depending on the `label_type`, it delegates to the
            corresponding method to fetch labels.
    """

    default_label_dtype = {
        "binary": tf.float32,
        "category_codes": tf.float32,
        "sparse_category_codes": tf.float32,
        "object_detection": tf.float32,
    }

    def __init__(self, label_type, num_classes=None, dtype=None):
        """
        Initializes the LabelManager with a specific label encoding type and,
        optionally, the number of classes.

        Args:
            - label_type (str): The type of label encoding to manage.
                Supported types are 'binary_codes',
                'category_codes','sparse_category_codes', and
                'object_detection'.
            - num_classes (int, optional): The total number of classes.
                Required if label_type is 'category_codes'.
            - dtype (tf.DType, optional): The data type of the label. If not
        """
        self.label_type = label_type
        self.num_classes = num_classes
        if label_type == "binary":
            self.encode_label = self.encode_binary_label
        elif label_type == "category_codes":
            self.encode_label = self.encode_categorical_label
        elif label_type == "sparse_category_codes":
            self.encode_label = self.encode_sparse_categorical_label
        elif label_type == "object_detection":
            self.encode_label = self.encode_object_detection_label
        else:
            msg = "The label type '{}' is not supported.".format(label_type)
            raise ValueError(msg)
        self.label_dtype = dtype or self.default_label_dtype.get(label_type, tf.int8)

    def encode_binary_label(self, sample):
        """
        Encodes binary label in sample into format suitable for binary
        classification.

        Args:
            - sample (dict): A dictionary containing the label data with key
                'label'.

        Returns:
            - tf.Tensor: A TensorFlow constant of the label in binary
                format.
        """
        try:
            label = int(sample["label"])
            if label not in [0, 1]:
                msg = "The 'label' key in sample is invalid for binary classification."
                raise ValueError(msg)
            label = tf.constant(label, dtype=self.label_dtype)
            return label
        except KeyError as e:
            msg = "The sample dictionary does not contain the key 'label'."
            raise KeyError(msg) from e
        except ValueError as e:
            msg = "The 'label' key in sample should be convertable to integer."
            raise ValueError(msg) from e

    def encode_categorical_label(self, sample):
        """
        Encodes catogircal label in sample into one-hot encoded format.

        Args:
            - sample (dict): A dictionary containing the label data with key
                'label'.

        Returns:
            - tf.Tensor: A one-hot encoded TensorFlow constant of the label.
        """
        try:
            label = int(sample["label"])
            label = tf.constant(label, dtype=tf.int8)
            label = tf.one_hot(label, self.num_classes)
            label = tf.cast(label, self.label_dtype)
            return label
        except KeyError as e:
            msg = "The sample dictionary does not contain the key 'label'."
            raise KeyError(msg) from e
        except ValueError as e:
            msg = "The 'label' key in sample should be convertable to integer."
            raise ValueError(msg) from e
        # Except num_classes is invalid
        except tf.errors.OpError as e:
            msg = "Probably the number of classes is invalid."
            raise ValueError(msg) from e

    def encode_sparse_categorical_label(self, sample):
        """
        Encodes categorical label in sample into sparse format suitable for
        sparse categorical crossentropy loss.

        Args:
            - sample (dict): A dictionary containing the label data with key
                'label'.

        Returns:
            - tf.Tensor: A TensorFlow constant of the label in sparse
                format.
        """
        try:
            label = int(sample["label"])
            label = tf.constant(label, dtype=self.label_dtype)
            return label
        except KeyError as e:
            msg = "The sample dictionary does not contain the key 'label'."
            raise KeyError(msg) from e
        except ValueError as e:
            msg = "The 'label' key in sample should be convertable to integer."
            raise ValueError(msg) from e

    def encode_object_detection_label(self, sample):
        """
        Stub method for future implementation of object detection label
        encoding.

        Args:
            - sample (dict): A dictionary containing the label data.

        Raises:
            - NotImplementedError: Indicates that the method is not yet
                implemented.
        """
        msg = "Object Detection Labels are not yet implemented."
        raise NotImplementedError(msg)
