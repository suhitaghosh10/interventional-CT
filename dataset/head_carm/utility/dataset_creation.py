import os
import json
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D

from dataset.head_carm.utility.constants import JSON_PATH, \
    TRAIN_CONEBEAM_PROJECTIONS_PATH, NEEDLE_PROJECTIONS_PATH, \
    TRAIN_HELICAL_PRIOR_PATH, VALIDATION_RECORDS_13_PATH, \
    TEST_RECORDS_13_PATH, CARMH_GT_UPPER_99_PERCENTILE, CARM_DCT_MAX, \
    IMG_DIM_INP_2D, TEST_NUM
from utility.constants import AUGMENTATION_MAX_ANGLE, MAX_ANGLE_PRIOR, HANN, \
    TOTAL_PROJECTION_NUM, SPARSE_PROJECTION_NUM, WIDTH, HEIGHT, ANGLES, \
    PROJECTIONS, PIXEL_SPACING_U, PIXEL_SPACING_V, VOXEL_SPACING_X, \
    VOXEL_SPACING_Y, VOXEL_SPACING_Z, VOLUME_DEPTH, VOLUME_HEIGHT, \
    VOLUME_WIDTH, VOLUME
from utility.ct_utils import mu2hu, hu2mu, filter_sinogram_3d
from utility.ict_system import ArtisQSystem, DetectorBinning
from utility.utils import rotate, flip_rotate, flip, scale


def generate_test_dataset(path):
    # test set, unbatched
    file_paths = [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if filename.endswith('.tfrecord')
    ]
    test_dataset = tf.data.TFRecordDataset(file_paths)
    teds = test_dataset.map(_decode_validation_data)
    teds = teds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    teds = teds.unbatch()  # [h, w, 3]
    teds = teds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    teds = teds.map(_hu2normalized)
    teds = teds.prefetch(buffer_size=2)

    return teds


def generate_datasets(val_path, test_path, batch_size, buffer_size=1024):
    # load training subjects
    with open(JSON_PATH, 'r', encoding='utf-8') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    # load cone-beam projections of chest
    file_paths = [
        os.path.join(TRAIN_CONEBEAM_PROJECTIONS_PATH, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)  # ([w, h, 360], [3], [3])
    vol_ds = vol_ds.repeat()

    # load needle projections
    file_paths = [
        os.path.join(NEEDLE_PROJECTIONS_PATH, filename)
        for filename in os.listdir(NEEDLE_PROJECTIONS_PATH)
        if filename.endswith('.tfrecord')
    ]
    needle_dataset = tf.data.TFRecordDataset(file_paths)
    ndl_ds = needle_dataset.map(_decode_needle_projections)  # ([u, v, 360], [3], [3])
    ndl_ds = ndl_ds.map(_random_rotate)  # [u, v, 360]
    ndl_ds = ndl_ds.repeat()
    ndl_ds = ndl_ds.shuffle(buffer_size=buffer_size)

    # load prior helical scans
    file_paths = [
        os.path.join(TRAIN_HELICAL_PRIOR_PATH, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    prior_ds = tf.data.TFRecordDataset(file_paths)
    prior_ds = prior_ds.map(_decode_prior)
    prior_ds = prior_ds.map(hu2mu)
    prior_ds = prior_ds.repeat()

    # training set
    combined_ds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # (([u, v, 360], [3], [3]), [u, v, 360])
    combined_ds = combined_ds.map(_tensorize)  # [u, v, 360], [3], [3], [u, v, 360]
    combined_ds = combined_ds.map(
        lambda x0, x1, x2, y: tf.numpy_function(
            func=_modify_shape_to_z_fov,
            inp=[x0, x1, x2, y],
            Tout=[tf.float32, tf.float32, tf.int64, tf.float32],
        )
    )
    # generate the 3D reconstructions from cone-beam head and needle projections
    combined_ds = combined_ds.map(
        lambda x0, x1, x2, y: tf.numpy_function(
            func=_reconstruct_3d,
            inp=[x0, x1, x2, y],
            Tout=tf.float32,
        )
    )  # [w, h, d, 2]
    tds = combined_ds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
    del combined_ds, needle_dataset, vol_ds, volumes_dataset

    tds = tf.data.Dataset.zip((tds, prior_ds))  # ([d, h, w, 2], [d, h, w, 1])
    tds = tds.map(
        lambda x, y: tf.numpy_function(
            func=_equalize_z_dimensions,
            inp=[x, y],
            Tout=[tf.float32, tf.float32],
        ))
    tds = tds.map(create_gt_from_tensors)
    tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(mu2hu)
    tds = tds.map(augmentation)  # ([2, h, w, 1], [h, w, 1])
    tds = tds.map(_hu2normalized)
    tds = tds.shuffle(buffer_size=buffer_size)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=2)

    # validation set
    file_paths = [
        os.path.join(val_path, filename)
        for filename in os.listdir(val_path)
        if filename.endswith('.tfrecord')
    ]
    val_dataset = tf.data.TFRecordDataset(file_paths)
    vds = val_dataset.map(_decode_validation_data)
    # vds = vds.map(lambda x, _y, _z: x)  # [w, h, d, 3]
    vds = vds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    vds = vds.unbatch()  # [h, w, 3]
    vds = vds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    vds = vds.map(_hu2normalized)
    vds = vds.batch(batch_size=batch_size)
    vds = vds.prefetch(buffer_size=2)

    # test set, unbatched
    teds = generate_test_dataset(test_path)

    return tds, vds, teds


def generate_datasets_wo_prior(val_path, test_path, batch_size, buffer_size=1024):
    # load training subjects
    with open(JSON_PATH, 'r', encoding='utf-8') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    # load con-beam projections of head
    file_paths = [
        os.path.join(TRAIN_CONEBEAM_PROJECTIONS_PATH, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)  # ([w, h, 360], [3], [3])
    vol_ds = vol_ds.repeat()

    # load needle projections
    file_paths = [
        os.path.join(NEEDLE_PROJECTIONS_PATH, filename)
        for filename in os.listdir(NEEDLE_PROJECTIONS_PATH)
        if filename.endswith('.tfrecord')
    ]
    needle_dataset = tf.data.TFRecordDataset(file_paths)
    ndl_ds = needle_dataset.map(_decode_needle_projections)  # ([u, v, 360], [3], [3])
    ndl_ds = ndl_ds.map(_random_rotate)  # [u, v, 360]
    ndl_ds = ndl_ds.repeat()
    ndl_ds = ndl_ds.shuffle(buffer_size=buffer_size)

    # training set
    tds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # (([u, v, 360], [3], [3]), [u, v, 360])
    tds = tds.map(_tensorize)
    # generate the 3D reconstructions from cone-beam head and needle projections
    tds = tds.map(
        lambda x0, x1, x2, y: tf.numpy_function(func=_reconstruct_3d, inp=[x0, x1, x2, y, True], Tout=tf.float32),
    )  # [w, h, d, 2]
    tds = tds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
    del needle_dataset, vol_ds, volumes_dataset

    # tds = tf.data.Dataset.zip((tds, prior_ds))  # ([d, h, w, 2], [d, h, w, 1])
    # tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(augmentation)  # ([2, h, w, 1], [h, w, 1])
    tds = tds.map(_hu2normalized)
    tds = tds.shuffle(buffer_size=buffer_size)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=2)

    # validation set
    file_paths = [
        os.path.join(val_path, filename)
        for filename in os.listdir(val_path)
        if filename.endswith('.tfrecord')
    ]
    val_dataset = tf.data.TFRecordDataset(file_paths)
    vds = val_dataset.map(_decode_validation_data_wo_prior)
    vds = vds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    vds = vds.unbatch()  # [h, w, 3]
    vds = vds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    vds = vds.map(_hu2normalized)
    vds = vds.batch(batch_size=batch_size)
    vds = vds.prefetch(buffer_size=2)

    # test set, unbatched
    file_paths = [
        os.path.join(test_path, filename)
        for filename in os.listdir(test_path)
        if filename.endswith('.tfrecord')
    ]
    test_dataset = tf.data.TFRecordDataset(file_paths)
    teds = test_dataset.map(_decode_validation_data_wo_prior)
    teds = teds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    teds = teds.unbatch()  # [h, w, 3]
    # teds = teds.shuffle(buffer_size=1024)
    teds = teds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    teds = teds.map(_hu2normalized)
    teds = teds.prefetch(buffer_size=2)

    return tds, vds, teds


# in/out: [u, v, 360], [3], [3], [u, v, 360]
def _modify_shape_to_z_fov(vol_projections: np.ndarray,
                           voxel_spacing: np.ndarray,
                           volume_shape: np.ndarray,
                           needle_projections: np.ndarray) \
                           -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    field_of_view = 200.  # FOV in z direction, 200mm
    num_fov_slices = int(field_of_view//voxel_spacing[2])
    num_fov_slices += (num_fov_slices - volume_shape[0]) % 2
    volume_shape[0] = min(volume_shape[0], num_fov_slices)
    return vol_projections, voxel_spacing, volume_shape, needle_projections


def _equalize_z_dimensions(reco_volume: tf.Tensor,
                           prior_volume: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if prior_volume.shape[0] == reco_volume.shape[0]:
        return reco_volume, prior_volume
    zdiff = (prior_volume.shape[0] - reco_volume.shape[0])//2
    return reco_volume, prior_volume[zdiff:-zdiff]


def create_gt_from_tensors(recos: tf.Tensor,
                           prior: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    reco_sparse, needle_full = recos[..., 0], recos[..., 1]
    full_reco = needle_full + prior[..., 0]
    return tf.stack([reco_sparse, full_reco], -1), prior


def _hu2normalized(tensor0: tf.Tensor, tensor1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return hu2mu(tensor0)/hu2mu(CARMH_GT_UPPER_99_PERCENTILE), \
        hu2mu(tensor1)/hu2mu(CARMH_GT_UPPER_99_PERCENTILE)


def _normalise(tensor: tf.Tensor):
    tensor = tf.clip_by_value(tensor, clip_value_min=0, clip_value_max=CARM_DCT_MAX)
    return tensor/CARM_DCT_MAX


def _hu2dctnormalized(tensor0: tf.Tensor, tensor1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return _normalise(tensor0), _normalise(tensor1)


def augmentation(input_tensor: tf.Tensor):
    sparse_reco = input_tensor[..., 0:1]
    prior = input_tensor[..., 2:3]
    full_reco = input_tensor[..., 1:2]

    sparse_reco = tf.cast(sparse_reco, dtype=tf.float32)
    prior = tf.cast(prior, dtype=tf.float32)
    full_reco = tf.cast(full_reco, dtype=tf.float32)

    # 0-none 1-rotate 2-scale 3-flip 4-fliprotate
    rand_num = tf.random.uniform(shape=[], minval=0, maxval=5, dtype=tf.int32, seed=10)
    angle = tf.random.uniform(shape=[], minval=-AUGMENTATION_MAX_ANGLE, maxval=AUGMENTATION_MAX_ANGLE,
                              dtype=tf.float32, seed=10)
    prior_angle = tf.random.uniform(shape=[], minval=-MAX_ANGLE_PRIOR, maxval=MAX_ANGLE_PRIOR,
                                    dtype=tf.float32, seed=10)
    scale_ratio = tf.random.uniform(shape=[], minval=0.8, maxval=1.0, dtype=tf.float32, seed=10)

    if tf.equal(rand_num, tf.constant(1)):
        sparse_reco, prior = rotate(sparse_reco, angle), rotate(prior, angle)
        full_reco = rotate(full_reco, angle)
    if tf.equal(rand_num, tf.constant(2)):
        sparse_reco, prior = scale(sparse_reco, scale_ratio), scale(prior, scale_ratio)
        full_reco = scale(full_reco, scale_ratio)
    if tf.equal(rand_num, tf.constant(3)):
        sparse_reco, prior = flip(sparse_reco), flip(prior)
        full_reco = flip(full_reco)
    if tf.equal(rand_num, tf.constant(4)):
        sparse_reco, prior = flip_rotate(sparse_reco, angle), flip_rotate(prior, angle)
        full_reco = flip_rotate(full_reco, angle)

    prior = rotate(prior, prior_angle)

    return tf.stack((sparse_reco, prior)), full_reco


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

    with torch.inference_mode():
        projections_t = torch.from_numpy(projections.transpose()).float().cuda()
        projections_t = projections_t[None, None, ...]
        filtered_projections_t = filter_sinogram_3d(projections_t, HANN)
        reco_t = radon.backprojection(filtered_projections_t)
        reco_t = reco_t*det_spacing_v/src_det_dist*src_dist  # cone beam correction
        # reco_t = mu2hu(reco_t, 0.02)
        reco = reco_t[0, 0].cpu()
        torch.cuda.empty_cache()
    return reco.numpy().transpose()


def _reconstruct_3d(vol_projections, voxel_size, volume_shape, ndl_projections, wo_prior=False):
    full_radon = create_radon(TOTAL_PROJECTION_NUM)
    sparse_radon = create_radon(SPARSE_PROJECTION_NUM)

    num_sparse_projections = len(sparse_radon.angles)
    voxel_dims = (IMG_DIM_INP_2D[0], IMG_DIM_INP_2D[1], volume_shape[0])

    # create interventional projections
    vol_ndl_projections = vol_projections + ndl_projections

    # create reconstruction of interventional volume w/ all projections
    full_needle = reconstruct_volume_from_projections(
         ndl_projections, full_radon, voxel_dims, voxel_size)

    # create reconstruction of interventional volume w/ sparse projections
    sparse_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections[...,
        ::(vol_ndl_projections.shape[-1] // num_sparse_projections + num_sparse_projections % 2)],
        sparse_radon, voxel_dims, voxel_size)
    if wo_prior:
        return np.stack((sparse_with_needle, full_needle, sparse_with_needle), -1)
    return np.stack((sparse_with_needle, full_needle), -1)


def _tensorize(vol_example: Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
               ndl_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return vol_example[0], tf.stack(vol_example[1]), tf.stack(vol_example[2]), ndl_example


def _get_volume(vol_example: Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
                ndl_example: tf.Tensor):
    return vol_example[0]


def _decode_vol_projections(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            PROJECTIONS: tf.io.FixedLenFeature([], tf.string),
            PIXEL_SPACING_U: tf.io.FixedLenFeature([], tf.float32),
            PIXEL_SPACING_V: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_X: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_Y: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_Z: tf.io.FixedLenFeature([], tf.float32),
            VOLUME_DEPTH: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_WIDTH: tf.io.FixedLenFeature([], tf.int64),
            WIDTH: tf.io.FixedLenFeature([], tf.int64),
            HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            ANGLES: tf.io.FixedLenFeature([], tf.int64),
        })

    projections = tf.io.decode_raw(feature[PROJECTIONS], tf.float32)
    width = feature[WIDTH]
    height = feature[HEIGHT]
    angles = feature[ANGLES]
    voxel_spacing = (
        feature[VOXEL_SPACING_X],
        feature[VOXEL_SPACING_Y],
        feature[VOXEL_SPACING_Z],
    )
    volume_shape = (
        feature[VOLUME_DEPTH],
        feature[VOLUME_HEIGHT],
        feature[VOLUME_WIDTH],
    )

    return tf.reshape(projections, (width, height, angles)), voxel_spacing, volume_shape


def _decode_needle_projections(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            PROJECTIONS: tf.io.FixedLenFeature([], tf.string),
            PIXEL_SPACING_U: tf.io.FixedLenFeature([], tf.float32),
            PIXEL_SPACING_V: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_X: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_Y: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_Z: tf.io.FixedLenFeature([], tf.float32),
            VOLUME_DEPTH: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_WIDTH: tf.io.FixedLenFeature([], tf.int64),
            WIDTH: tf.io.FixedLenFeature([], tf.int64),
            HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            ANGLES: tf.io.FixedLenFeature([], tf.int64),
        })

    projections = tf.io.decode_raw(feature[PROJECTIONS], tf.float32)
    width = feature[WIDTH]
    height = feature[HEIGHT]
    angles = feature[ANGLES]

    return tf.reshape(projections, (width, height, angles))


def _random_rotate(image: tf.Tensor):
    return tf.roll(
        image,
        tf.random.uniform(
            shape=[],
            minval=0,
            # maxval=ndl_projections.get_shape()[-1],  # 360
            maxval=360,
            dtype=tf.int32,
            seed=10),
        -1)


def _decode_prior(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            VOLUME: tf.io.FixedLenFeature([], tf.string),
            VOXEL_SPACING_X: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_Y: tf.io.FixedLenFeature([], tf.float32),
            VOXEL_SPACING_Z: tf.io.FixedLenFeature([], tf.float32),
            VOLUME_DEPTH: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            VOLUME_WIDTH: tf.io.FixedLenFeature([], tf.int64),
        })

    volume = tf.io.decode_raw(feature[VOLUME], tf.float32)
    depth = feature[VOLUME_DEPTH]
    height = feature[VOLUME_HEIGHT]
    width = feature[VOLUME_WIDTH]
    # voxel_spacing = (
    #     feature[VOXEL_SPACING_X],
    #     feature[VOXEL_SPACING_Y],
    #     feature[VOXEL_SPACING_Z],
    # )

    # return tf.reshape(volume, (depth, height, width, 1)), voxel_spacing
    return tf.reshape(volume, (depth, height, width, 1))


def _decode_validation_data(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'reco_tensor': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'subject_file': tf.io.FixedLenFeature([], tf.string),
            'needle_file': tf.io.FixedLenFeature([], tf.string),
        })

    reco_tensor = tf.io.decode_raw(feature['reco_tensor'], tf.float32)
    depth = feature['depth']
    height = feature['height']
    width = feature['width']
    # subject_file = feature['subject_file']
    # needle_file = feature['needle_file']

    tensor = tf.reshape(reco_tensor, (depth, height, width, 3))
    tensor = tf.transpose(tensor, perm=(2, 1, 0, 3))

    return tensor


def _decode_validation_data_wo_prior(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'reco_tensor': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'subject_file': tf.io.FixedLenFeature([], tf.string),
            'needle_file': tf.io.FixedLenFeature([], tf.string),
        })

    reco_tensor = tf.io.decode_raw(feature['reco_tensor'], tf.float32)
    depth = feature['depth']
    height = feature['height']
    width = feature['width']

    tensor = tf.reshape(reco_tensor, (depth, height, width, 3))
    tensor = tf.transpose(tensor, perm=(2, 1, 0, 3))
    tensor = tf.stack((tensor[..., 0], tensor[..., 1], tensor[..., 0]), axis=-1) # remove the helical prior and replace it with nosiy img

    return tensor


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


def _tuplelize(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.expand_dims(tf.stack([tensor[..., 0], tensor[..., 2]]), axis=3), tensor[..., 1:2]

# def test_validation_data():
#     import tqdm
#     file_paths = [
#         os.path.join(VALIDATION_RECORDS_PATH, filename)
#         for filename in os.listdir(VALIDATION_RECORDS_PATH)
#         if filename.endswith('.tfrecord')
#     ]
#
#     val_dataset = tf.data.TFRecordDataset(file_paths)
#     vds = val_dataset.map(_decode_validation_data)
#     vds = vds.map(lambda x, _y, _z: x)  # [d, h, w, 3]
#     vds = vds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
#     vds = vds.unbatch()  # [h, w, 3]
#     vds = vds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
#     vds = vds.map(_hu2normalized)
#     vds = vds.batch(batch_size=32)
#     vds = vds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#     from time import time
#     ds_iter = iter(vds)
#     input_data, output_data = next(ds_iter)
#     print(input_data.shape, output_data.shape)
#     t0 = time()
#     num_iterations: int = VAL_NUM // 32
#     for _ in tqdm(range(num_iterations - 1)):
#         _ = next(ds_iter)
#     print(f'{(time() - t0)/num_iterations}s per batch')


def test_training_set():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as err:
            print(err)

    tds, _, _ = generate_datasets(
        VALIDATION_RECORDS_13_PATH,
        TEST_RECORDS_13_PATH, 32, 1)
    from time import time
    ds_iter = iter(tds)
    input_data, output_data = next(ds_iter)
    print(input_data.shape, output_data.shape)
    from matplotlib import pyplot as plt
    plt.imshow(input_data[31, 0, ..., 0], vmin=0, vmax=.7)
    plt.figure()
    plt.imshow(input_data[31, 1, ..., 0], vmin=0, vmax=.7)
    plt.figure()
    plt.imshow(output_data[31, ..., 0], vmin=0, vmax=.7)
    plt.show()
    print(input_data.shape, output_data.shape)

    from utility.utils import dct_and_pixelwise_mse
    dct_loss_fn = dct_and_pixelwise_mse(IMG_DIM_INP_2D)
    loss = dct_loss_fn(output_data, output_data)
    print(f'{loss=}')

    t0 = time()
    num_iterations: int = TEST_NUM // 32
    from tqdm import tqdm
    for _ in tqdm(range(num_iterations - 1)):
        _ = next(ds_iter)
    print(f'{(time() - t0)/num_iterations}s per batch')

    # ([d, h, w, 2], ([d, h, w, 1], [3]))


if __name__ == '__main__':
    test_training_set()
