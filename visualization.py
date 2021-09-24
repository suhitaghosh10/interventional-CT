import json
import os
from typing import Tuple, Union

import cv2
import nibabel as nib
import numpy as np
import tensorflow as tf

from dataset.head_carm.models.prior_unet import unet
from dataset.head_carm.utility.dataset_creation import \
    _reconstruct_3D, _decode_prior, _hu2normalized, _tuplelize
from dataset.head_carm.utility.constants import IMG_DIM_INP_2D, JSON_PATH
from utility.constants import CHKPOINT_NAME
from utility.ct_utils import create_projections
from utility.weight_norm import AdamWithWeightnorm
from utility.utils import ssim, psnr, mse, load_nrrd, load_nifty


def generate_testset(batch_size: Union[int, None],
                     subject_idx: int,
                     needle_idx: int,
                     roll: int,
                     center: Tuple[float, float, float]):
    volume_dir: str = '/mnt/nvme2/mayoclinic/Head/high_dose'
    needle_dir: str = '/home/phernst/Documents/git/ictdl/needles/'

    with open(JSON_PATH, 'r') as file_handle:
        json_dict = json.load(file_handle)
        test_subjects = json_dict['test_subjects']

    prior_file = os.path.join(volume_dir, 'priors', test_subjects[subject_idx]) + '.tfrecord'
    volume_file = os.path.join(volume_dir, test_subjects[subject_idx]) + '.nrrd'
    needle_file = os.path.join(needle_dir, [
        f for f in os.listdir(needle_dir)
        if 'upperPart' not in f and 'lowerPart' not in f][needle_idx])

    volume_projections, _, voxel_size, volume_shape = create_projections(volume_file, load_nrrd)
    needle_projections, _, _, _ = create_projections(needle_file, load_nifty, center=center)
    needle_projections = np.roll(needle_projections, roll, -1)

    reco = _reconstruct_3D(volume_projections, voxel_size, volume_shape, needle_projections)  # [w, h, d, 2]
    print(reco.shape)

    prior_ds = tf.data.TFRecordDataset([prior_file])
    prior_ds = prior_ds.map(_decode_prior)

    tds = tf.data.Dataset.from_tensor_slices(reco[None, ...])
    tds = tds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
    tds = tf.data.Dataset.zip((tds, prior_ds))  # ([d, h, w, 2], [d, h, w, 1])
    tds = tds.map(lambda x, y: tf.concat([x, y], axis=3))  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(_tuplelize)  # ([2, h, w, 1], [h, w, 1])
    tds = tds.map(_hu2normalized)
    if batch_size is not None:
        tds = tds.batch(batch_size=batch_size)
    return tds


def get_model(path: str):
    unet_cls = unet()

    act = tf.keras.layers.LeakyReLU(alpha=0.2)
    model = unet_cls.build_model(d=8, act=act)
    optimizer = AdamWithWeightnorm(learning_rate=1e-4)

    model.compile(optimizer=optimizer,
                  run_eagerly=False,
                  loss=mse(IMG_DIM_INP_2D),
                  metrics=[ssim(IMG_DIM_INP_2D),
                           psnr(IMG_DIM_INP_2D),
                           mse(IMG_DIM_INP_2D)
                           ])
    CHKPNT_PATH = os.path.join(path)
    model.load_weights(os.path.join(CHKPNT_PATH, CHKPOINT_NAME)).expect_partial()
    return model


def windowing(image: np.ndarray, vmin: float, vmax: float):
    return np.clip((image - vmin)/(vmax - vmin), 0, 1)


def extract_input_slices(subject_idx: int,
                         needle_idx: int,
                         roll: int,
                         center: Tuple[float, float, float],
                         xy_pos: int,
                         xz_pos: int):
    test_ds = generate_testset(None, subject_idx, needle_idx, roll, center)
    elements_in = np.concatenate([e for e, _ in test_ds], axis=-1).transpose(0, 2, 1, 3)
    elements_gt = np.concatenate([e for _, e in test_ds], axis=-1).transpose(1, 0, 2)

    save_path = os.path.join(
        'experiments', 'predictions',
        f'{subject_idx}_{needle_idx}')
    os.makedirs(save_path, exist_ok=True)

    # coronal slices
    sparse_cor = elements_in[0, :, xz_pos, :]
    prior_cor = elements_in[1, :, xz_pos, :]
    gt_cor = elements_gt[:, xz_pos, :]

    cv2.imwrite(
        os.path.join(save_path, 'sparse_cor.png'),
        windowing(sparse_cor, .3689, .3915)*255)
    cv2.imwrite(
        os.path.join(save_path, 'prior_cor.png'),
        windowing(prior_cor, .3689, .3915)*255)
    cv2.imwrite(
        os.path.join(save_path, 'gt_cor.png'),
        windowing(gt_cor, .3689, .3915)*255)

    # axial slices
    sparse_axial = elements_in[0, :, :, xy_pos]
    prior_axial = elements_in[1, :, :, xy_pos]
    gt_axial = elements_gt[:, :, xy_pos]

    cv2.imwrite(
        os.path.join(save_path, 'sparse_axial.png'),
        windowing(sparse_axial, .3689, .3915)*255)
    cv2.imwrite(
        os.path.join(save_path, 'prior_axial.png'),
        windowing(prior_axial, .3689, .3915)*255)
    cv2.imwrite(
        os.path.join(save_path, 'gt_axial.png'),
        windowing(gt_axial, .3689, .3915)*255)


def generate_predictions(subject_idx: int,
                         needle_idx: int,
                         roll: int,
                         center: Tuple[float, float, float],
                         xy_pos: int,
                         xz_pos: int):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # expt = Experiment()
    # act = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    save_path = os.path.join(
        'experiments', 'predictions',
        f'{subject_idx}_{needle_idx}')
    os.makedirs(save_path, exist_ok=True)
    test_ds = generate_testset(64, subject_idx, needle_idx, roll, center)

    model_names = ['Unet_Prior_needle_MSE_D8Lr0.0001_S1342',
                   'Unet_Prior_needle_DCT_MSE_D8Lr0.0001_S1342',
                   'Unet_Prior_needle_LoG0.1_MSE10.0_D8Lr0.0001_S1342',
                   'Unet_woPrior_needle_MSE_D8Lr0.0001_S1342',
                   'Unet_Prior_needle_MSSIM_D8Lr0.0001_S1342',
                   'Unet_Prior_needle_SSIM_D8Lr0.0001_S1342']
    for name in model_names:
        print(name)
        model = get_model(os.path.join('experiments', 'suhi', name, 'chkpnt'))
        results = model.evaluate(test_ds, batch_size=128)
        print('ssim psnr mse', format(results[1], ".5f"), format(results[2], ".5f"), format(results[3], ".3e"))

        arr = model.predict(test_ds, batch_size=64)[..., 0]
        np.save(os.path.join(save_path, name+'.npy'), arr)
        img = nib.Nifti1Image(arr.transpose(), np.eye(4))
        nib.save(img, os.path.join(save_path, name+'.nii.gz'))
        cv2.imwrite(
            os.path.join(save_path, name+'_axial.png'),
            windowing(arr.transpose()[:, :, xy_pos], .3689, .3915)*255)
        cv2.imwrite(
            os.path.join(save_path, name+'_coronal.png'),
            windowing(arr.transpose()[:, xz_pos, :], .3689, .3915)*255)


if __name__ == '__main__':
    # generate_predictions(subject_idx=0, needle_idx=0, roll=0, center=(0, 0, -50), xy_pos=145, xz_pos=208)
    generate_predictions(subject_idx=1, needle_idx=1, roll=180, center=(0, 0, 0), xy_pos=278, xz_pos=173)
    # extract_input_slices(subject_idx=0, needle_idx=0, roll=0, center=(0, 0, -50), xy_pos=145, xz_pos=208)
    # extract_input_slices(subject_idx=1, needle_idx=1, roll=180, center=(0, 0, 0), xy_pos=278, xz_pos=173)

# test[0], needle[0], center=(0, 0, -50), 146/469, xz 209, window: [.3689, .3915]
# test[1], needle[1], center=(0, 0, 0), 279/493, xz 174, roll 180
