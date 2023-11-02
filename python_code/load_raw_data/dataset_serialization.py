import os

import tensorflow as tf
from python_code.utils.pcb_visualization import PCBVisualizerforTF

# ------ Save to Record -------------------------------------------------------------
def _bytes_feature(value):
    # Helper function to create a TensorFlow Feature containing a list of bytes (required to serialize to a protocol buffer format).
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(image, label):
    # Serialize an image-label pair into a TFRecord-compatible Example object.
    feature = {
        'image': _bytes_feature(tf.io.encode_jpeg(image).numpy()),
        'label': _bytes_feature(tf.io.serialize_tensor(label).numpy())
    }
    sample_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample_proto.SerializeToString()

def save_tfrecord_from_file(dataset, filepath):
    # Saves a tf.data.Dataset object to a TFRecord file.
    
    if not os.path.exists(os.path.dirname(filepath)):
        raise FileNotFoundError(f"Directory '{os.path.dirname(filepath)}' does not exist.")
    
    with tf.io.TFRecordWriter(filepath) as writer:
        for image, label in dataset:
            example = serialize_sample(image, label)
            writer.write(example)

# ------ Load to Record -------------------------------------------------------------
def parse_tfrecord(sample_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(sample_proto, feature_description)
    image = tf.io.decode_jpeg(sample['image'], channels=3)
    label = tf.io.parse_tensor(sample['label'], out_type=tf.int8)
    return image, label

def load_tfrecord_from_file(filepath):
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"tfrecord '{filepath}' does not exist.")
    
    raw_dataset = tf.data.TFRecordDataset(filepath)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    parsed_dataset = parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Allows parallel processing of multiple items.
    return parsed_dataset

  
