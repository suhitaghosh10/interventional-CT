from dataset.head_carm.utility.constants import *
from utility.constants import *
import json
import os
import tensorflow as tf

# can be used for validation and test records
def _decode_validation_data(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
        })

    depth = feature['depth']
    height = feature['height']
    width = feature['width']

    return (depth, height, width)

# count test or validation records
def count_test_tfrecords(path):
    file_paths = [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if filename.endswith('.tfrecord')
    ]
    val_dataset = tf.data.TFRecordDataset(file_paths)
    vds = val_dataset.map(_decode_validation_data)
    z_slices_num = 0
    for shape in vds:
        z, _, _ = shape
        print(z.numpy())
        z_slices_num += z.numpy()

    return z_slices_num


# used for train
def decode_vol_projections(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            VOLUME_DEPTH: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_WIDTH: tf.io.FixedLenFeature([], tf.int64),
            WIDTH: tf.io.FixedLenFeature([], tf.int64),
            HEIGHT: tf.io.FixedLenFeature([], tf.int64)
        })

    return feature[VOLUME_DEPTH]

#count train
def count_train_tfrecords():
    # load training subjects
    with open(JSON_PATH, 'r') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    # load con-beam projections of head
    file_paths = [
        os.path.join(TRAIN_CONEBEAM_PROJECTIONS_PATH, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(decode_vol_projections)
    z_slices_num = 0
    for depth in vol_ds:
        z_slices_num += depth.numpy()
        print(depth.numpy())

    return z_slices_num

if __name__ == '__main__':
    count_test_tfrecords(VALIDATION_RECORDS_PATH)
    #print(count_train_tfrecords())
