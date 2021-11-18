import json
import os
import random

import tensorflow as tf
from tqdm import tqdm

from dataset.head_carm.utility.constants import *
from dataset.head_carm.utility.dataset_creation import _modify_shape_to_z_fov
from dataset.head_carm.utility.dataset_creation import _decode_vol_projections
from dataset.head_carm.utility.dataset_creation import _decode_validation_data
from utility.constants import *


# # can be used for validation and test records
# def _decode_validation_data(example_proto):
#     feature = tf.io.parse_single_example(
#         example_proto,
#         features={
#             'depth': tf.io.FixedLenFeature([], tf.int64),
#             'height': tf.io.FixedLenFeature([], tf.int64),
#             'width': tf.io.FixedLenFeature([], tf.int64),
#         })

#     depth = feature['depth']
#     height = feature['height']
#     width = feature['width']

#     return (depth, height, width)

# count test or validation records
def count_test_slices():
    # load cone beam projections of head
    file_paths = [
        os.path.join(TEST_RECORDS_13_PATH, f)
        for f in os.listdir(TEST_RECORDS_13_PATH)
    ]
    val_dataset = tf.data.TFRecordDataset(file_paths)
    vds = val_dataset.map(_decode_validation_data)
    z_slices_num = 0
    for element in vds:
        _, _, z, _ = element.shape
        print(z)
        z_slices_num += z

    print(z_slices_num)

    return z_slices_num


# count train
def count_train_slices():
    # load training subjects
    with open(JSON_PATH, 'r') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    # load cone beam projections of head
    file_paths = [
        os.path.join(TRAIN_CONEBEAM_PROJECTIONS_PATH, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)
    z_slices_num = 0
    for element in tqdm(vol_ds):
        _, voxel_spacing, volume_shape = element
        volume_shape = list(volume_shape)
        z_slices_num += _modify_shape_to_z_fov(
            None, voxel_spacing, volume_shape, None)[2][0]

    print(z_slices_num)

    return z_slices_num


def create_train_valid_test():
    all_subjects = [
        f[:-7] for f in os.listdir('/mnt/nvme2/lungs/lungs3d')
        if f.endswith('.nii.gz')
    ]
    random.shuffle(all_subjects)
    train_subjects = all_subjects[:int(0.7*len(all_subjects))]
    valid_subjects = all_subjects[int(0.7*len(all_subjects)):int(0.85*len(all_subjects))]
    test_subjects = all_subjects[int(0.85*len(all_subjects)):]
    with open('train_valid_test.json', 'w', newline='') as json_handle:
        json.dump({
            'train_subjects': train_subjects,
            'valid_subjects': valid_subjects,
            'test_subjects': test_subjects,
        }, json_handle)


if __name__ == '__main__':
    count_test_slices()
