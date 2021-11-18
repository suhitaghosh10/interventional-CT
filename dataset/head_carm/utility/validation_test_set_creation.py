import os
from os.path import join as pjoin
import json

import tensorflow as tf
from tqdm import tqdm

from dataset.head_carm.utility.constants import *
from dataset.head_carm.utility.dataset_creation import _tensorize, \
    _decode_needle_projections, _decode_prior, _decode_vol_projections, \
    _modify_shape_to_z_fov, _reconstruct_3D, _equalize_z_dimensions, \
    create_gt_from_tensors
from utility.constants import *
from utility.ct_utils import mu2hu, hu2mu
from utility.utils import _int64_feature, _bytes_feature


def generate_datasets(valid_test: str, out_path: str):
    assert valid_test in ['valid', 'test']

    example_path = pjoin(out_path, f'{SPARSE_PROJECTION_NUM}')
    os.makedirs(example_path, exist_ok=True)

    # load subjects
    with open(JSON_PATH, 'r') as file_handle:
        json_dict = json.load(file_handle)
        subjects = json_dict[f'{valid_test}_subjects']

    # load needles
    needles = [
        'Needle2_Pos1_11',
        'Needle2_Pos2_12',
        'Needle2_Pos3_13',
    ]

    for subject, needle in tqdm([(s, n) for s in subjects for n in needles]):
        # load cone-beam projections of chest
        file_paths = [pjoin(TRAIN_CONEBEAM_PROJECTIONS_PATH, f'{subject}.tfrecord')]
        volumes_dataset = tf.data.TFRecordDataset(file_paths)
        vol_ds = volumes_dataset.map(_decode_vol_projections)

        # load needle projections
        file_paths = [pjoin(NEEDLE_PROJECTIONS_PATH, f'{needle}.tfrecord')]
        needle_dataset = tf.data.TFRecordDataset(file_paths)
        ndl_ds = needle_dataset.map(_decode_needle_projections)

        # load prior helical scans
        file_paths = [pjoin(TRAIN_HELICAL_PRIOR_PATH, f'{subject}.tfrecord')]
        prior_ds = tf.data.TFRecordDataset(file_paths)
        prior_ds = prior_ds.map(_decode_prior)
        prior_ds = prior_ds.map(hu2mu)

        # training set
        combined_ds = tf.data.Dataset.zip((vol_ds, ndl_ds))
        combined_ds = combined_ds.map(_tensorize)
        combined_ds = combined_ds.map(
            lambda x0, x1, x2, y: tf.numpy_function(
                func=_modify_shape_to_z_fov,
                inp=[x0, x1, x2, y],
                Tout=[tf.float32, tf.float32, tf.int64, tf.float32],
            )
        )
        combined_ds = combined_ds.map(
            lambda x0, x1, x2, y: tf.numpy_function(
                func=_reconstruct_3D,
                inp=[x0, x1, x2, y],
                Tout=tf.float32,
            )
        )
        tds = combined_ds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))

        tds = tf.data.Dataset.zip((tds, prior_ds))
        tds = tds.map(
            lambda x, y: tf.numpy_function(
                func=_equalize_z_dimensions,
                inp=[x, y],
                Tout=[tf.float32, tf.float32],
            ))
        tds = tds.map(create_gt_from_tensors)
        tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
        tds = tds.map(mu2hu)

        data_tensor = next(iter(tds))
        # serialized_tensor = tf.io.serialize_tensor(data_tensor)
        feature_of_bytes = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data_tensor.numpy().flatten().tobytes()])
        )

        with tf.io.TFRecordWriter(pjoin(example_path, f'{subject}_{needle}.tfrecord')) as file_writer:
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                'reco_tensor': feature_of_bytes,
                'depth': _int64_feature(data_tensor.shape[0]),
                'height': _int64_feature(data_tensor.shape[1]),
                'width': _int64_feature(data_tensor.shape[2]),
                'subject_file': _bytes_feature(subject.encode('utf-8')),
                'needle_file': _bytes_feature(needle.encode('utf-8')),
            })).SerializeToString()
            file_writer.write(record_bytes)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    generate_datasets(
        'test',
        '/home/phernst/Documents/git/interventional-CT/ds_test'
    )
