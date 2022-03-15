import gc
import json
import os
import random
from typing import Callable

import tensorflow as tf
from tqdm import tqdm

from dataset.head_carm.utility.dataset_creation import _modify_shape_to_z_fov
from dataset.head_carm.utility.dataset_creation import _decode_vol_projections
from dataset.head_carm.utility.dataset_creation import _decode_validation_data
from dataset.head_carm.utility.constants import IMG_DIM_INP_2D, JSON_PATH, \
    TEST_RECORDS_13_PATH, TRAIN_CONEBEAM_PROJECTIONS_PATH
from utility.utils import dct_and_pixelwise_mae, dct_and_pixelwise_mse, \
    mae, multiscale_ssim_l2, mse, multiscale_ssim_l1, ssim_l2, \
    laplacian_of_gaussian_MAE


def set_available_gpus(cmd_args):
    gpu_id = cmd_args.gpu
    nb_gpus = len(gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return nb_gpus


def apply_tf_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as err:
            print(err)


class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def run_distributed(run_fn: Callable, *args):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # Open a strategy scope.
    with strategy.scope():
        run_fn(*args)


def needle_weighting(tensor: tf.Tensor) -> tf.Tensor:
    return tf.math.sigmoid(20*(tensor - .6))


def generate_loss_dict():
    return {
        'MSE': mse(IMG_DIM_INP_2D),
        'WMSE': mse(IMG_DIM_INP_2D, weight_fn=needle_weighting),
        'MSSIM_MSE': multiscale_ssim_l2(IMG_DIM_INP_2D),
        'MSSIM_WMSE': multiscale_ssim_l2(IMG_DIM_INP_2D, weight_fn=needle_weighting),
        'SSIM_MSE': ssim_l2(IMG_DIM_INP_2D),
        'SSIM_WMSE': ssim_l2(IMG_DIM_INP_2D, weight_fn=needle_weighting),
        'DCT_MSE': dct_and_pixelwise_mse(IMG_DIM_INP_2D),  # only eager mode
        # 'LoG_MSE': laplacian_of_gaussian_MSE(  # only eager mode
        #     IMG_DIM_INP_2D,
        #     mse_weight=10.,
        #     log_weight=0.1),
        'L1': mae(IMG_DIM_INP_2D),
        'MSSIM_L1': multiscale_ssim_l1(IMG_DIM_INP_2D),
        'DCT_L1': dct_and_pixelwise_mae(IMG_DIM_INP_2D),  # only eager mode
        'LoG_L1': laplacian_of_gaussian_MAE(IMG_DIM_INP_2D),  # only eager mode
    }


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
    with open(JSON_PATH, 'r', encoding='utf-8') as file_handle:
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
    with open('train_valid_test.json', 'w', newline='', encoding='utf-8') as json_handle:
        json.dump({
            'train_subjects': train_subjects,
            'valid_subjects': valid_subjects,
            'test_subjects': test_subjects,
        }, json_handle)


if __name__ == '__main__':
    count_test_slices()
