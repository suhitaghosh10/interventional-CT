import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D

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


def generate_datasets(data_path, batch_size=1, buffer_size=1024):
    file_paths = []
    for folder, subs, files in os.walk(os.path.join(data_path, TRAIN)):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    volumes_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    vol_ds = volumes_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [w, h, 360]
    vol_ds = vol_ds.repeat()
    vol_ds = vol_ds.shuffle(buffer_size=buffer_size)

    file_paths = []
    for folder, subs, files in os.walk(os.path.join(data_path, NEEDLE_PROJECTIONS)):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    needle_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    ndl_ds = needle_dataset.map(_decode_projections, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [u, v, 360]
    ndl_ds = ndl_ds.map(_random_rotate_needle, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [u, v, 360]
    ndl_ds = ndl_ds.repeat()
    ndl_ds = ndl_ds.shuffle(buffer_size=buffer_size)

    full_radon = create_radon(360)
    sparse_radon = create_radon(18)  # TODO
    def patched_reco_fn(example_proto):
        return _reconstruct_3D_poc(example_proto, full_radon, sparse_radon)

    # training set
    tds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # ([u, v, 360], [u, v, 360])
    tds = tds.map(patched_reco_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # [d, h, w, 3]
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
    teds = test_dataset.map(_decode_test_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    teds = teds.repeat()
    teds = teds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return tds, vds, teds


def _decode_projections(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'projections': tf.io.FixedLenFeature([], tf.string),
            'pixel_spacing_u': tf.io.FixedLenFeature([], tf.string),
            'pixel_spacing_v': tf.io.FixedLenFeature([], tf.string),
            'width': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.string),
            'angles': tf.io.FixedLenFeature([], tf.string),
        })

    projections = tf.io.decode_raw(feature['projections'], tf.float32)
    # pixel_spacing_u = tf.io.decode_raw(feature['pixel_spacing_u'], tf.float32)
    # pixel_spacing_v = tf.io.decode_raw(feature['pixel_spacing_v'], tf.float32)
    width = tf.io.decode_raw(feature['width'], tf.int64)
    height = tf.io.decode_raw(feature['height'], tf.int64)
    angles = tf.io.decode_raw(feature['angles'], tf.int64)

    return tf.reshape(projections, (width, height, angles))


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
    vol_projections, ndl_projections = example_proto
    vol_projections = vol_projections.numpy()
    ndl_projections = ndl_projections.numpy()

    num_sparse_projections = len(sparse_radon.angles)
    voxel_dims = (384, 384, 500)  # TODO
    voxel_size = (0.48828125, 0.48828125, 0.3515625)  # TODO

    # create reconstruction of prior volume
    prior_reco = reconstruct_volume_from_projections(
        vol_projections, full_radon, voxel_dims, voxel_size)
    prior_reco = tf.convert_to_tensor(prior_reco, tf.float32)

    # create interventional projections
    vol_ndl_projections = vol_projections + ndl_projections

    # create reconstruction of interventional volume w/ all projections
    full_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections, full_radon, voxel_dims, voxel_size)
    full_with_needle = tf.convert_to_tensor(full_with_needle, tf.float32)

    # create reconstruction of interventional volume w/ sparse projections
    sparse_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections[..., ::vol_ndl_projections.shape[-1]//num_sparse_projections],
        sparse_radon, voxel_dims, voxel_size)
    sparse_with_needle = tf.convert_to_tensor(sparse_with_needle, tf.float32)
    # return [sparse_with_needle, prior_reco], full_with_needle
    return tf.stack((sparse_with_needle, prior_reco, full_with_needle), -1)


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
            depth=voxel_dims[0],
            height=voxel_dims[1],
            width=voxel_dims[2],
            voxel_size=voxel_size)

    det_spacing_v = radon.projections.cfg.det_spacing_v
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
