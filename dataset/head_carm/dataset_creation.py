import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D
from tqdm import tqdm

from dataset.head_carm.constants import *
from utility.ct_utils import mu2hu, filter_sinogram_3d
from utility.utils import augment_prior
from utility.den_utils import read_den_volume
from utility.ict_system import ArtisQSystem, DetectorBinning

CARMHEAD_2D_TFRECORDS_TRAIN = 'carmhead.tfrecords.train'
CARMHEAD_2D_TFRECORDS_VAL = 'carmhead.tfrecords.val'
CARMHEAD_2D_TFRECORDS_TEST = 'carmhead.tfrecords.test'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _tensorize(vol_example: Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
               ndl_example: tf.Tensor):
    return vol_example[0], tf.stack(vol_example[1]), tf.stack(vol_example[2]), ndl_example


def test_reconstruction():
    file_paths = ['/mnt/nvme2/mayoclinic/Head/high_dose_projections/N005c.tfrecord']
    volumes_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    vol_ds = volumes_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    file_paths = ['/home/phernst/Documents/git/ictdl/needle_projections/Needle2_Pos2_12.tfrecord']
    needle_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    ndl_ds = needle_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ndl_ds = ndl_ds.map(lambda x, _y, _z: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    full_radon = create_radon(360)
    sparse_radon = create_radon(18)  # TODO
    def patched_reco_fn(vol0, vol1, vol2, ndl_example):
        return _reconstruct_3D_poc(((vol0, vol1, vol2), ndl_example), full_radon, sparse_radon)

    tds = tf.data.Dataset.zip((vol_ds, ndl_ds))
    tds = tds.map(_tensorize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tds = tds.map(lambda x0, x1, x2, y: tf.numpy_function(func=patched_reco_fn, inp=[x0, x1, x2, y], Tout=tf.float32),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(augment_prior)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    tds = tds.shuffle(buffer_size=100)
    tds = tds.batch(3)
    tds = tds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iter_tds = iter(tds)
    _ = next(iter_tds)
    from time import time
    t0 = time()
    num_iterations: int = 100
    for _ in range(num_iterations):
        _ = next(iter_tds)
    print(f'{(time() - t0)/num_iterations}s per batch')


def generate_datasets(data_path, batch_size=1, buffer_size=1024):
    file_paths = []
    for folder, _, files in os.walk(os.path.join(data_path, HEAD_PROJECTIONS)):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    volumes_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    vol_ds = volumes_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # ([w, h, 360], [3], [3])
    vol_ds = vol_ds.repeat()
    vol_ds = vol_ds.shuffle(buffer_size=buffer_size)

    file_paths = []
    for folder, _, files in os.walk(os.path.join(data_path, NEEDLE_PROJECTIONS)):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    needle_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    ndl_ds = needle_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # ([u, v, 360], [3], [3])
    ndl_ds = ndl_ds.map(lambda x: x[0], num_parallel_calls=tf.data.experimental.AUTOTUNE) # [u, v, 360]
    ndl_ds = ndl_ds.map(_random_rotate_needle, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [u, v, 360]
    ndl_ds = ndl_ds.repeat()
    ndl_ds = ndl_ds.shuffle(buffer_size=buffer_size)

    full_radon = create_radon(360)
    sparse_radon = create_radon(18)  # TODO
    def patched_reco_fn(vol0, vol1, vol2, ndl_example):
        return _reconstruct_3D_poc(((vol0, vol1, vol2), ndl_example), full_radon, sparse_radon)

    # training set
    tds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # (([u, v, 360], [3], [3]), [u, v, 360])
    tds = tds.map(_tensorize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tds = tds.map(lambda x0, x1, x2, y: tf.numpy_function(func=patched_reco_fn, inp=[x0, x1, x2, y], Tout=tf.float32),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(augment_prior)  # ([[h, w], [h, w]], [h, w])
    tds = tds.shuffle(buffer_size=buffer_size)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # validation set # TODO
    val_dataset = tf.data.TFRecordDataset(os.path.join(data_path , VAL, CARMHEAD_2D_TFRECORDS_VAL))
    vds = val_dataset.map(_decode_validation_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    vds = vds.batch(batch_size=batch_size)
    vds = vds.repeat()
    vds = vds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # test set, unbatched # TODO
    test_dataset = tf.data.TFRecordDataset(os.path.join(data_path, TEST, CARMHEAD_2D_TFRECORDS_TEST))
    teds = test_dataset.map(_decode_validation_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    teds = teds.repeat()
    teds = teds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return tds, vds, teds


def _decode_projections(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'projections': tf.io.FixedLenFeature([], tf.string),
            'pixel_spacing_u': tf.io.FixedLenFeature([], tf.float32),
            'pixel_spacing_v': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_x': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_y': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_z': tf.io.FixedLenFeature([], tf.float32),
            'volume_depth': tf.io.FixedLenFeature([], tf.int64),
            'volume_height': tf.io.FixedLenFeature([], tf.int64),
            'volume_width': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'angles': tf.io.FixedLenFeature([], tf.int64),
        })

    projections = tf.io.decode_raw(feature['projections'], tf.float32)
    # pixel_spacing_u = tf.io.decode_raw(feature['pixel_spacing_u'], tf.float32)
    # pixel_spacing_v = tf.io.decode_raw(feature['pixel_spacing_v'], tf.float32)
    width = feature['width']
    height = feature['height']
    angles = feature['angles']
    voxel_spacing = (
        feature['voxel_spacing_x'],
        feature['voxel_spacing_y'],
        feature['voxel_spacing_z'],
    )
    volume_shape = (
        feature['volume_depth'],
        feature['volume_height'],
        feature['volume_width'],
    )

    return tf.reshape(projections, (width, height, angles)), \
        voxel_spacing, volume_shape


def _random_rotate_needle(ndl_projections: tf.Tensor):
    return tf.roll(
        ndl_projections,
        tf.random.uniform(
            shape=[],
            minval=0,
            maxval=ndl_projections.get_shape()[-1],  # 360
            dtype=tf.int32,
            seed=10),
        -1)


def _reconstruct_3D_poc(example_proto, full_radon: ConeBeam, sparse_radon: ConeBeam):
    (vol_projections, voxel_size, volume_shape), ndl_projections = example_proto

    num_sparse_projections = len(sparse_radon.angles)
    voxel_dims = (384, 384, volume_shape[0])

    # create reconstruction of prior volume
    prior_reco = reconstruct_volume_from_projections(
        vol_projections, full_radon, voxel_dims, voxel_size)

    # create interventional projections
    vol_ndl_projections = vol_projections + ndl_projections

    # create reconstruction of interventional volume w/ all projections
    full_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections, full_radon, voxel_dims, voxel_size)

    # create reconstruction of interventional volume w/ sparse projections
    sparse_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections[..., ::vol_ndl_projections.shape[-1]//num_sparse_projections],
        sparse_radon, voxel_dims, voxel_size)

    return np.stack((sparse_with_needle, prior_reco, full_with_needle), -1)


def create_radon(num_views: int) -> ConeBeam:
    ct_system = ArtisQSystem(DetectorBinning.BINNING4x4)
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False, dtype=np.float32)
    src_dist = ct_system.carm_span*4/6
    det_dist = ct_system.carm_span*2/6
    # src_det_dist = src_dist + det_dist
    det_spacing_v = ct_system.pixel_dims[1]
    return ConeBeam(
        det_count_u=ct_system.nb_pixels[0],
        angles=angles,
        src_dist=src_dist,
        det_dist=det_dist,
        det_count_v=ct_system.nb_pixels[1],
        det_spacing_u=ct_system.pixel_dims[0],
        det_spacing_v=det_spacing_v,
        pitch=0.0,
        base_z=0.0,
    )


def reconstruct_volume_from_projections(projections: np.ndarray, radon: ConeBeam,
                                        voxel_dims: Tuple[int, int, int],
                                        voxel_size: Tuple[float, float, float]) \
                                        -> np.ndarray:
    """
    returns reconstruction of input projections in HU, [z, y, x]
    """
    assert len(radon.angles) == projections.shape[-1]
    radon.volume = Volume3D(
            depth=voxel_dims[2],
            height=voxel_dims[1],
            width=voxel_dims[0],
            voxel_size=voxel_size)

    det_spacing_v = radon.projection.cfg.det_spacing_v
    src_dist = radon.projection.cfg.s_dist
    det_dist = radon.projection.cfg.d_dist
    src_det_dist = src_dist + det_dist

    projections_t = torch.from_numpy(projections.transpose()).float().cuda()
    projections_t = projections_t[None, None, ...]
    reco_t = radon.backprojection(filter_sinogram_3d(projections_t, 'hann'))
    reco_t = reco_t*det_spacing_v/src_det_dist*src_dist  # cone beam correction
    reco_t = mu2hu(reco_t, 0.02)
    return reco_t[0, 0].cpu().numpy().transpose()


def generate_tf_records(data_path, save_path, create_tf_record=[True, True, True]):
    z_start = 80
    z_end = 395
    print('start creating tf-records')
    if create_tf_record[0]:
        train_shards_path = os.path.join(save_path, TRAIN)
        print('create shards')
        with tf.io.TFRecordWriter(os.path.join(save_path, CARMHEAD_2D_TFRECORDS_TRAIN)) as writer:
            path = os.path.join(data_path, TRAIN)
            files = os.listdir(path)
            for filename in files:
                print(filename)
                vol = read_den_volume(os.path.join(data_path, TRAIN, filename, 'vol.den'), block=4,
                                 type=np.dtype('<f4')).astype(np.float16)
                vol_15 = read_den_volume(os.path.join(data_path, TRAIN, filename, 'vol_15.den'), block=4,
                                    type=np.dtype('<f4')).astype(np.float16)
                # slices = vol.shape[0]

                for slice in range(z_start, z_end):
                    image = np.clip(vol[:, :, slice], a_min=CARMH_IMG_LOW_5_PERCENTILE,
                                    a_max=CARMH_IMG_UPPER_99_PERCENTILE)
                    image = (image - CARMH_IMG_LOW_5_PERCENTILE) / (
                            CARMH_IMG_UPPER_99_PERCENTILE - CARMH_IMG_LOW_5_PERCENTILE)
                    # print(image.shapes)
                    image = np.reshape(image, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))
                    # np.save(os.path.join(save_path,'img.npy'), image)
                    annotation = np.clip(vol_15[:, :, slice], a_min=CARMH_GT_LOW_5_PERCENTILE,
                                         a_max=CARMH_GT_UPPER_99_PERCENTILE)
                    annotation = (annotation - CARMH_GT_LOW_5_PERCENTILE) / (
                            CARMH_GT_UPPER_99_PERCENTILE - CARMH_GT_LOW_5_PERCENTILE)
                    annotation = np.reshape(annotation, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))
                    # np.save(os.path.join(save_path, 'ann.npy'), annotation)
                    img_raw = image.tostring()
                    annotation_raw = annotation.tostring()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'img': _bytes_feature(img_raw),
                        'name': _bytes_feature(tf.compat.as_bytes(filename + '_s' + str(slice))),
                        'gt': _bytes_feature(annotation_raw)}))
                    writer.write(example.SerializeToString())
        # # creating shards
        print('training tf record created')
        shards_num = 25
        raw_dataset = tf.data.TFRecordDataset(os.path.join(save_path, CARMHEAD_2D_TFRECORDS_TRAIN))
        for shard_idx in range(shards_num):
            writer = tf.data.experimental.TFRecordWriter(f"{train_shards_path}/w-{shard_idx}.tfrecord")
            writer.write(raw_dataset.shard(shards_num, shard_idx))
        print('created tf-record shards for training set')

    if create_tf_record[1]:
        with tf.io.TFRecordWriter(os.path.join(save_path, VAL, CARMHEAD_2D_TFRECORDS_VAL)) as writer:
            path = os.path.join(data_path, VAL)
            files = os.listdir(path)
            print(files)
            for filename in files:
                print(filename)
                vol = read_den_volume(os.path.join(data_path, VAL, filename, 'vol.den'), block=4,
                                 type=np.dtype('<f4')).astype(np.float16)
                vol_15 = read_den_volume(os.path.join(data_path, VAL, filename, 'vol_15.den'), block=4,
                                    type=np.dtype('<f4')).astype(np.float16)

                for slice in range(z_start, z_end):
                    image = np.clip(vol[:, :, slice], a_min=CARMH_IMG_LOW_5_PERCENTILE,
                                    a_max=CARMH_IMG_UPPER_99_PERCENTILE)
                    image = (image - CARMH_IMG_LOW_5_PERCENTILE) / (
                            CARMH_IMG_UPPER_99_PERCENTILE - CARMH_IMG_LOW_5_PERCENTILE)
                    # print(image.shapes)
                    image = np.reshape(image, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))

                    annotation = np.clip(vol_15[:, :, slice], a_min=CARMH_GT_LOW_5_PERCENTILE,
                                         a_max=CARMH_GT_UPPER_99_PERCENTILE)
                    annotation = (annotation - CARMH_GT_LOW_5_PERCENTILE) / (
                            CARMH_GT_UPPER_99_PERCENTILE - CARMH_GT_LOW_5_PERCENTILE)
                    annotation = np.reshape(annotation, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))

                    img_raw = image.tostring()
                    annotation_raw = annotation.tostring()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'img': _bytes_feature(img_raw),
                        'name': _bytes_feature(tf.compat.as_bytes(filename + '_s' + str(slice))),
                        'gt': _bytes_feature(annotation_raw)}))
                    writer.write(example.SerializeToString())
        print('created tf-records for val set')

    if create_tf_record[2]:
        with tf.io.TFRecordWriter(os.path.join(save_path, TEST, CARMHEAD_2D_TFRECORDS_TEST)) as writer:
            path = os.path.join(data_path, TEST)
            files = os.listdir(path)
            for filename in files:
                print(filename)
                vol = read_den_volume(os.path.join(data_path, TEST, filename, 'vol.den'), block=4,
                                 type=np.dtype('<f4')).astype(np.float16)
                vol_15 = read_den_volume(os.path.join(data_path, TEST, filename, 'vol_15.den'), block=4,
                                    type=np.dtype('<f4')).astype(np.float16)

                for slice in range(z_start, z_end):
                    image = np.clip(vol[:, :, slice], a_min=CARMH_IMG_LOW_5_PERCENTILE,
                                    a_max=CARMH_IMG_UPPER_99_PERCENTILE)
                    image = (image - CARMH_IMG_LOW_5_PERCENTILE) / (
                            CARMH_IMG_UPPER_99_PERCENTILE - CARMH_IMG_LOW_5_PERCENTILE)
                    # print(image.shapes)
                    image = np.reshape(image, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))

                    annotation = np.clip(vol_15[:, :, slice], a_min=CARMH_GT_LOW_5_PERCENTILE,
                                         a_max=CARMH_GT_UPPER_99_PERCENTILE)
                    annotation = (annotation - CARMH_GT_LOW_5_PERCENTILE) / (
                            CARMH_GT_UPPER_99_PERCENTILE - CARMH_GT_LOW_5_PERCENTILE)
                    annotation = np.reshape(annotation, (IMG_DIM_ORIG_2D[0], IMG_DIM_ORIG_2D[1], 1))

                    img_raw = image.tostring()
                    annotation_raw = annotation.tostring()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'img': _bytes_feature(img_raw),
                        'name': _bytes_feature(tf.compat.as_bytes(filename + '_s' + str(slice))),
                        'gt': _bytes_feature(annotation_raw)}))
                    writer.write(example.SerializeToString())
    print('created tf-records for test set')


def generate_train_valid_test_split(subject_directory: str):
    all_subjects = [f[:f.index('.')] for f in os.listdir(subject_directory)]
    random.shuffle(all_subjects)
    train_subjects = all_subjects[:35]
    valid_subjects = all_subjects[35:35+8]
    test_subjects = all_subjects[-7:]
    with open('train_valid_test.json', 'w') as file_handle:
        json.dump({
            'train_subjects': train_subjects,
            'valid_subjects': valid_subjects,
            'test_subjects': test_subjects,
        }, file_handle)


def generate_validation_record(subject_list: List[str], out_path: str,
                               subject_dir: Optional[str] = None,
                               needle_dir: Optional[str] = None):
    subject_dir = HEAD_PROJECTIONS if subject_dir is None else subject_dir
    needle_dir = NEEDLE_PROJECTIONS if needle_dir is None else needle_dir
    needle_files = [n for n in os.listdir(needle_dir) if n.endswith('.tfrecord')]
    subject_needle_combinations = [
        (x+'.tfrecord', y) for x in subject_list for y in needle_files
    ]

    full_radon = create_radon(360)
    sparse_radon = create_radon(18)  # TODO
    def patched_reco_fn(vol0, vol1, vol2, ndl_example):
        return _reconstruct_3D_poc(((vol0, vol1, vol2), ndl_example), full_radon, sparse_radon)

    for subject_file, needle_file in tqdm(subject_needle_combinations):
        out_file = f'{subject_file[:subject_file.index(".")]}_' \
            f'{needle_file[:needle_file.index(".")]}.tfrecord'
        with tf.io.TFRecordWriter(os.path.join(out_path, out_file)) as writer:
            subject_ds = tf.data.TFRecordDataset([os.path.join(subject_dir, subject_file)])
            subject_ds = subject_ds.map(_decode_projections)

            needle_ds = tf.data.TFRecordDataset([os.path.join(needle_dir, needle_file)])
            needle_ds = needle_ds.map(_decode_projections)
            needle_ds = needle_ds.map(lambda x, _y, _z: x)

            # training set
            tds = tf.data.Dataset.zip((subject_ds, needle_ds))
            tds = tds.map(_tensorize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            tds = tds.map(lambda x0, x1, x2, y: tf.numpy_function(func=patched_reco_fn, inp=[x0, x1, x2, y], Tout=tf.float32),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [d, h, w, 3]
            reco_tensor = next(iter(tds)).numpy()
            reco_shape = reco_tensor.shape

            example = tf.train.Example(features=tf.train.Features(feature={
                'reco_tensor': _bytes_feature(value=reco_tensor.tobytes()),
                'depth': _int64_feature(reco_shape[0]),
                'height': _int64_feature(reco_shape[1]),
                'width': _int64_feature(reco_shape[2]),
                'subject_file':  _bytes_feature(subject_file.encode('utf-8')),
                'needle_file':  _bytes_feature(needle_file.encode('utf-8')),
            }))
            writer.write(example.SerializeToString())


def create_validation_set_record():
    with open('train_valid_test.json', 'r') as file_handle:
        valid_subjects = json.load(file_handle)['valid_subjects']
    generate_validation_record(
        valid_subjects,
        'ds_validation',
        subject_dir='/mnt/nvme2/mayoclinic/Head/high_dose_projections',
        needle_dir='/home/phernst/Documents/git/ictdl/needle_projections')


def create_test_set_record():
    with open('train_valid_test.json', 'r') as file_handle:
        test_subjects = json.load(file_handle)['test_subjects']
    generate_validation_record(
        test_subjects,
        'ds_test',
        subject_dir='/mnt/nvme2/mayoclinic/Head/high_dose_projections',
        needle_dir='/home/phernst/Documents/git/ictdl/needle_projections')


if __name__ == '__main__':
    create_validation_set_record()
