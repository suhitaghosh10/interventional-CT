import json
import os
import random
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import tensorflow as tf
import torch
from torch_radon import ConeBeam
from torch_radon.volumes import Volume3D
from tqdm import tqdm
from utility.utils import _bytes_feature, _int64_feature

from dataset.head_carm.utility.constants import *
from utility.ct_utils import mu2hu, hu2mu, filter_sinogram_3d
from utility.utils import augment_prior
from utility.ict_system import ArtisQSystem, DetectorBinning

CARMHEAD_2D_TFRECORDS_TRAIN = 'carmhead.tfrecords.train'
CARMHEAD_2D_TFRECORDS_VAL = 'carmhead.tfrecords.val'
CARMHEAD_2D_TFRECORDS_TEST = 'carmhead.tfrecords.test'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

SPARSE_PROJECTION_NUM = 18
TOTAL_PROJECTION_NUM = 360


def _tensorize(vol_example: Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
               ndl_example: tf.Tensor):
    return vol_example[0], tf.stack(vol_example[1]), tf.stack(vol_example[2]), ndl_example


def _tuplelize(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.expand_dims(tf.stack([tensor[..., 0], tensor[..., 2]]), axis=3), tensor[..., 1:2]


def _hu2normalized(tensor0: tf.Tensor, tensor1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return hu2mu(tensor0, .02)/hu2mu(CARMH_GT_UPPER_99_PERCENTILE, .02), \
            hu2mu(tensor1, .02)/hu2mu(CARMH_GT_UPPER_99_PERCENTILE, .02)


def count_training_slices():
    with open('train_valid_test.json', 'r') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    file_paths = [
        os.path.join(HEAD_PROJECTIONS, f'{tr}.tfrecord')
        for tr in train_subjects
    ]

    def get_num_slices(_x, _y, volume_shape):
        return volume_shape[0]

    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)
    vol_ds = vol_ds.map(get_num_slices)
    all_num_slices = vol_ds.reduce(np.int64(0), lambda x, y: x + y).numpy()
    print(all_num_slices)  # 17180


def count_valid_slices():
    file_paths = [
        os.path.join(VALIDATION_RECORDS, tr)
        for tr in os.listdir(VALIDATION_RECORDS)
    ]

    def get_num_slices(_x, _y, volume_shape):
        return volume_shape[0]

    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)
    vol_ds = vol_ds.map(get_num_slices)
    all_num_slices = vol_ds.reduce(np.int64(0), lambda x, y: x + y).numpy()
    print(all_num_slices)  # 3877


def test_reconstruction(batch_size: int, buffer_size: int):
    with open('train_valid_test.json', 'r') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    file_paths = [
        os.path.join(HEAD_PROJECTIONS, f'{tr}.tfrecord')
        for tr in train_subjects
    ][:1]
    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)  # ([w, h, 360], [3], [3])
    vol_ds = vol_ds.repeat()

    file_paths = [
        os.path.join(NEEDLE_PROJECTIONS, filename)
        for filename in os.listdir(NEEDLE_PROJECTIONS)
        if filename.endswith('.tfrecord')
    ][:1]
    print(file_paths)
    needle_dataset = tf.data.TFRecordDataset(file_paths)
    ndl_ds = needle_dataset.map(_decode_needle_projections)  # ([u, v, 360], [3], [3])
    ndl_ds = ndl_ds.repeat()

    file_paths = [
        os.path.join(HEAD_PRIORS, f'{tr}.tfrecord')
        for tr in train_subjects
    ][:1]
    print(file_paths)
    prior_ds = tf.data.TFRecordDataset(file_paths)
    prior_ds = prior_ds.map(_decode_prior)
    prior_ds = prior_ds.repeat()

    # training set
    tds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # (([u, v, 360], [3], [3]), [u, v, 360])
    tds = tds.map(_tensorize)
    tds = tds.map(
        lambda x0, x1, x2, y: tf.numpy_function(func=_reconstruct_3D_poc, inp=[x0, x1, x2, y], Tout=tf.float32),
    )  # [w, h, d, 3]
    tds = tds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
    tds = tf.data.Dataset.zip((tds, prior_ds))  # ([d, h, w, 2], [d, h, w, 1])
    tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(augment_prior)  # ([2, h, w, 1], [h, w, 1])
    tds = tds.shuffle(buffer_size=100)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=buffer_size)

    iter_tds = iter(tds)
    for _ in tqdm(range(30)):
        sparse_input, full_input = next(iter_tds)

    img = nib.Nifti1Image(sparse_input[:, 0, ..., 0].numpy().transpose(), np.eye(4))
    nib.save(img, 'sparse_with_needle.nii.gz')
    img = nib.Nifti1Image(sparse_input[:, 1, ..., 0].numpy().transpose(), np.eye(4))
    nib.save(img, 'prior_reco.nii.gz')
    img = nib.Nifti1Image(full_input[..., 0].numpy().transpose(), np.eye(4))
    nib.save(img, 'full_with_needle.nii.gz')

    # from time import time
    # t0 = time()
    # num_iterations: int = 2000
    # for _ in tqdm(range(num_iterations)):
    #     _ = next(iter_tds)
    # print(f'{(time() - t0)/num_iterations}s per batch')


def generate_datasets(batch_size=1, buffer_size=1024):
    with open('train_valid_test.json', 'r') as file_handle:
        json_dict = json.load(file_handle)
        train_subjects = json_dict['train_subjects']

    file_paths = [
        os.path.join(HEAD_PROJECTIONS, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    volumes_dataset = tf.data.TFRecordDataset(file_paths)
    vol_ds = volumes_dataset.map(_decode_vol_projections)  # ([w, h, 360], [3], [3])
    vol_ds = vol_ds.repeat()

    file_paths = [
        os.path.join(NEEDLE_PROJECTIONS, filename)
        for filename in os.listdir(NEEDLE_PROJECTIONS)
        if filename.endswith('.tfrecord')
    ]
    needle_dataset = tf.data.TFRecordDataset(file_paths)
    ndl_ds = needle_dataset.map(_decode_needle_projections)  # ([u, v, 360], [3], [3])
    ndl_ds = ndl_ds.map(_random_rotate_needle)  # [u, v, 360]
    ndl_ds = ndl_ds.repeat()
    ndl_ds = ndl_ds.shuffle(buffer_size=buffer_size)

    file_paths = [
        os.path.join(HEAD_PRIORS, f'{tr}.tfrecord')
        for tr in train_subjects
    ]
    prior_ds = tf.data.TFRecordDataset(file_paths)
    prior_ds = prior_ds.map(_decode_prior)
    prior_ds = prior_ds.repeat()

    # training set
    tds = tf.data.Dataset.zip((vol_ds, ndl_ds))  # (([u, v, 360], [3], [3]), [u, v, 360])
    tds = tds.map(_tensorize)
    tds = tds.map(
        lambda x0, x1, x2, y: tf.numpy_function(func=_reconstruct_3D_poc, inp=[x0, x1, x2, y], Tout=tf.float32),
    )  # [w, h, d, 2]
    tds = tds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
    tds = tf.data.Dataset.zip((tds, prior_ds))  # ([d, h, w, 2], [d, h, w, 1])
    tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(augment_prior)  # ([2, h, w, 1], [h, w, 1])
    tds = tds.map(_hu2normalized)
    tds = tds.shuffle(buffer_size=buffer_size)
    tds = tds.batch(batch_size)
    tds = tds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # validation set
    file_paths = [
        os.path.join(VALIDATION_RECORDS, filename)
        for filename in os.listdir(VALIDATION_RECORDS)
        if filename.endswith('.tfrecord')
    ]
    val_dataset = tf.data.TFRecordDataset(file_paths)
    vds = val_dataset.map(_decode_validation_data)
    vds = vds.map(lambda x, _y, _z: x)  # [w, h, d, 3]
    vds = vds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    vds = vds.unbatch()  # [h, w, 3]
    vds = vds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    vds = vds.map(_hu2normalized)
    vds = vds.batch(batch_size=batch_size)
    vds = vds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # test set, unbatched
    file_paths = [
        os.path.join(TEST_RECORDS, filename)
        for filename in os.listdir(TEST_RECORDS)
        if filename.endswith('.tfrecord')
    ]
    test_dataset = tf.data.TFRecordDataset(file_paths)
    teds = test_dataset.map(_decode_validation_data)
    teds = teds.map(lambda x, _y, _z: x)  # [w, h, d, 3]
    teds = teds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    teds = teds.unbatch()  # [h, w, 3]
    teds = teds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    teds = teds.map(_hu2normalized)
    teds = teds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return tds, vds, teds


def _decode_vol_projections(example_proto):
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

    return tf.reshape(projections, (width, height, angles)), voxel_spacing, volume_shape
    #return tf.reshape(projections, (width, height, angles))

def _decode_needle_projections(example_proto):
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

    #return tf.reshape(projections, (width, height, angles)), voxel_spacing, volume_shape
    return tf.reshape(projections, (width, height, angles))


def _decode_prior(example_proto):
    feature = tf.io.parse_single_example(
        example_proto,
        features={
            'volume': tf.io.FixedLenFeature([], tf.string),
            'voxel_spacing_x': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_y': tf.io.FixedLenFeature([], tf.float32),
            'voxel_spacing_z': tf.io.FixedLenFeature([], tf.float32),
            'volume_depth': tf.io.FixedLenFeature([], tf.int64),
            'volume_height': tf.io.FixedLenFeature([], tf.int64),
            'volume_width': tf.io.FixedLenFeature([], tf.int64),
        })

    volume = tf.io.decode_raw(feature['volume'], tf.float32)
    depth = feature['volume_depth']
    height = feature['volume_height']
    width = feature['volume_width']
    voxel_spacing = (
        feature['voxel_spacing_x'],
        feature['voxel_spacing_y'],
        feature['voxel_spacing_z'],
    )

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
    subject_file = feature['subject_file']
    needle_file = feature['needle_file']

    tensor = tf.reshape(reco_tensor, (depth, height, width, 3))
    tensor = tf.transpose(tensor, perm=(2, 1, 0, 3))

    return tensor, subject_file, needle_file


def test_validation_data():
    file_paths = [
        os.path.join(VALIDATION_RECORDS, filename)
        for filename in os.listdir(VALIDATION_RECORDS)
        if filename.endswith('.tfrecord')
    ]

    val_dataset = tf.data.TFRecordDataset(file_paths)
    vds = val_dataset.map(_decode_validation_data)
    vds = vds.map(lambda x, _y, _z: x)  # [d, h, w, 3]
    vds = vds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))
    vds = vds.unbatch()  # [h, w, 3]
    vds = vds.map(_tuplelize)  # ([[h, w, 1], [h, w, 1]], [h, w, 1])
    vds = vds.map(_hu2normalized)
    vds = vds.batch(batch_size=32)
    vds = vds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    from time import time
    ds_iter = iter(vds)
    input_data, output_data = next(ds_iter)
    print(input_data.shape, output_data.shape)
    t0 = time()
    num_iterations: int = VAL_NUM // 32
    for _ in tqdm(range(num_iterations - 1)):
        _ = next(ds_iter)
    print(f'{(time() - t0)/num_iterations}s per batch')


def _random_rotate_needle(ndl_projections: tf.Tensor):
    return tf.roll(
        ndl_projections,
        tf.random.uniform(
            shape=[],
            minval=0,
            #maxval=ndl_projections.get_shape()[-1],  # 360
            maxval=360,
            dtype=tf.int32,
            seed=10),
        -1)


def _reconstruct_3D_poc(vol_projections, voxel_size, volume_shape, ndl_projections):
    full_radon = create_radon(TOTAL_PROJECTION_NUM)
    sparse_radon = create_radon(SPARSE_PROJECTION_NUM)  # TODO

    num_sparse_projections = len(sparse_radon.angles)
    voxel_dims = (384, 384, volume_shape[0])

    # create interventional projections
    vol_ndl_projections = vol_projections + ndl_projections

    # create reconstruction of interventional volume w/ all projections
    full_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections, full_radon, voxel_dims, voxel_size)

    # create reconstruction of interventional volume w/ sparse projections
    sparse_with_needle = reconstruct_volume_from_projections(
        vol_ndl_projections[..., ::vol_ndl_projections.shape[-1]//num_sparse_projections],
        sparse_radon, voxel_dims, voxel_size)

    return np.stack((sparse_with_needle, full_with_needle), -1)


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
    reco = reco_t[0, 0].cpu()
    torch.cuda.empty_cache()
    return reco.numpy().transpose()


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
    prior_dir = HEAD_PRIORS
    needle_dir = NEEDLE_PROJECTIONS if needle_dir is None else needle_dir
    needle_files = [n for n in os.listdir(needle_dir) if n.endswith('.tfrecord')]
    subject_needle_combinations = [
        (x+'.tfrecord', y) for x in subject_list for y in needle_files
    ]

    for subject_file, needle_file in tqdm(subject_needle_combinations):
        out_file = f'{subject_file[:subject_file.index(".")]}_' \
            f'{needle_file[:needle_file.index(".")]}.tfrecord'
        with tf.io.TFRecordWriter(os.path.join(out_path, out_file)) as writer:
            subject_ds = tf.data.TFRecordDataset([os.path.join(subject_dir, subject_file)])
            subject_ds = subject_ds.map(_decode_vol_projections)

            prior_ds = tf.data.TFRecordDataset([os.path.join(prior_dir, subject_file)])
            prior_ds = prior_ds.map(_decode_prior)  # [d, h, w, 1]

            needle_ds = tf.data.TFRecordDataset([os.path.join(needle_dir, needle_file)])
            needle_ds = needle_ds.map(_decode_needle_projections)

            # training set
            tds = tf.data.Dataset.zip((subject_ds, needle_ds))
            tds = tds.map(_tensorize)
            tds = tds.map(
                lambda x0, x1, x2, y: tf.numpy_function(func=_reconstruct_3D_poc, inp=[x0, x1, x2, y], Tout=tf.float32),
            )  # [w, h, d, 2]
            tds = tds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
            tds = tf.data.Dataset.zip((tds, prior_ds))
            tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))
            reco_tensor = next(iter(tds)).numpy()
            reco_shape = reco_tensor.shape

            example = tf.train.Example(features=tf.train.Features(feature={
                'reco_tensor': _bytes_feature(value=reco_tensor.tobytes()),
                'depth': _int64_feature(reco_shape[0]),
                'height': _int64_feature(reco_shape[1]),
                'width': _int64_feature(reco_shape[2]),
                'subject_file': _bytes_feature(subject_file.encode('utf-8')),
                'needle_file': _bytes_feature(needle_file.encode('utf-8')),
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    # test_reconstruction(16, 2)
    test_validation_data()
