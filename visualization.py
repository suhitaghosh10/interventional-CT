import json
import os
from typing import Tuple, Union

import cv2
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skssim
import tensorflow as tf

from dataset.head_carm.models.prior_unet import UNet
from dataset.head_carm.utility.dataset_creation import \
    _reconstruct_3d, _decode_prior, _hu2normalized, _tuplelize
from dataset.head_carm.utility.constants import IMG_DIM_INP_2D, JSON_PATH, \
    CARMH_GT_UPPER_99_PERCENTILE
from utility.constants import CHKPOINT_NAME
from utility.ct_utils import create_projections, mu2hu, hu2mu
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

    reco = _reconstruct_3d(volume_projections, voxel_size, volume_shape, needle_projections)  # [w, h, d, 2]
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


def generate_testset_wo_prior(batch_size: Union[int, None],
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

    reco = _reconstruct_3d(volume_projections, voxel_size, volume_shape, needle_projections)  # [w, h, d, 2]

    print(reco.shape)

    tds = tf.data.Dataset.from_tensor_slices(reco[None, ...])
    tds = tds.map(lambda x: tf.transpose(x, (2, 1, 0, 3)))  # [d, h, w, 2]
    tds = tds.map(lambda x: tf.concat([x, x[..., 0:1]], axis=3))  # [d, h, w, 3]
    tds = tds.unbatch()  # [h, w, 3]
    tds = tds.map(_tuplelize)  # ([2, h, w, 1], [h, w, 1])
    tds = tds.map(_hu2normalized)
    if batch_size is not None:
        tds = tds.batch(batch_size=batch_size)
    return tds


def create_figure1_images():
    volume_dir: str = '/mnt/nvme2/mayoclinic/Head/high_dose'
    needle_dir: str = '/home/phernst/Documents/git/ictdl/needles/'
    with open(JSON_PATH, 'r') as file_handle:
        json_dict = json.load(file_handle)
        test_subjects = json_dict['test_subjects']

    subject_idx: int = 0
    needle_idx: int = 1
    volume_file = os.path.join(volume_dir, test_subjects[subject_idx]) + '.nrrd'
    needle_file = os.path.join(needle_dir, [
        f for f in os.listdir(needle_dir)
        if 'upperPart' not in f and 'lowerPart' not in f][needle_idx])
    print(needle_file)

    test_ds = generate_testset(None, 0, 0, 0, (0, 0, -50))
    elements_in = np.concatenate([e for e, _ in test_ds], axis=-1).transpose(0, 2, 1, 3)
    elements_gt = np.concatenate([e for _, e in test_ds], axis=-1).transpose(1, 0, 2)

    volume_projections, _, voxel_size, volume_shape = create_projections(volume_file, load_nrrd)
    needle_projections, _, _, _ = create_projections(needle_file, load_nifty, center=(0, 0, 0))
    needle_projections = np.zeros_like(needle_projections)

    reco = _reconstruct_3d(volume_projections, voxel_size, volume_shape, needle_projections)  # [w, h, d, 2]
    reco = _hu2normalized(reco, reco)[0]

    sparse_needle_vol = elements_in[0].transpose()
    gt_needle_vol = elements_gt.transpose()
    gt_vol = reco.transpose()[1]
    sparse_vol = reco.transpose()[0]

    # slice 264, [.3519, .4149]
    # axial_idx = 264
    # window_full = (.3519, .4149)
    # window_sparse = (.2090, .6444)

    # slice 149
    axial_idx = 149
    window_sparse = (.2090, .6444)
    window_full = window_sparse  # (.3519, .4149)

    cv2.imwrite(
        os.path.join('visualization', 'sparse_needle.png'),
        windowing(sparse_needle_vol[axial_idx], window_sparse[0], window_sparse[1])*255)
    cv2.imwrite(
        os.path.join('visualization', 'gt_needle.png'),
        windowing(gt_needle_vol[axial_idx], window_full[0], window_full[1])*255)
    cv2.imwrite(
        os.path.join('visualization', 'gt.png'),
        windowing(gt_vol[axial_idx], window_full[0], window_full[1])*255)
    cv2.imwrite(
        os.path.join('visualization', 'sparse.png'),
        windowing(sparse_vol[axial_idx], window_sparse[0], window_sparse[1])*255)


def get_model(path: str):
    unet_cls = UNet()

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
        os.path.join(save_path, 'sparse_coronal.png'),
        windowing(sparse_cor, 0, 1)*255)
    cv2.imwrite(
        os.path.join(save_path, 'prior_coronal.png'),
        windowing(prior_cor, 0, 1)*255)
    cv2.imwrite(
        os.path.join(save_path, 'gt_coronal.png'),
        windowing(gt_cor, 0, 1)*255)

    np.save(os.path.join(save_path, 'sparse_coronal.npy'), sparse_cor)
    np.save(os.path.join(save_path, 'prior_coronal.npy'), prior_cor)
    np.save(os.path.join(save_path, 'gt_coronal.npy'), gt_cor)

    # axial slices
    sparse_axial = elements_in[0, :, :, xy_pos]
    prior_axial = elements_in[1, :, :, xy_pos]
    gt_axial = elements_gt[:, :, xy_pos]

    cv2.imwrite(
        os.path.join(save_path, 'sparse_axial.png'),
        windowing(sparse_axial, 0, 1)*255)
    cv2.imwrite(
        os.path.join(save_path, 'prior_axial.png'),
        windowing(prior_axial, 0, 1)*255)
    cv2.imwrite(
        os.path.join(save_path, 'gt_axial.png'),
        windowing(gt_axial, 0, 1)*255)

    np.save(os.path.join(save_path, 'sparse_axial.npy'), sparse_axial)
    np.save(os.path.join(save_path, 'prior_axial.npy'), prior_axial)
    np.save(os.path.join(save_path, 'gt_axial.npy'), gt_axial)


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
    test_ds = generate_testset(None, subject_idx, needle_idx, roll, center)
    elements_gt = np.concatenate([e for _, e in test_ds], axis=-1).transpose(1, 0, 2)
    elements_sparse = np.concatenate([e for e, _ in test_ds], axis=-1).transpose(0, 2, 1, 3)[0]
    np.save(os.path.join(save_path, 'gt_subject.npy'), elements_gt)
    np.save(os.path.join(save_path, 'sparse_subject.npy'), elements_sparse)
    test_ds = generate_testset(64, subject_idx, needle_idx, roll, center)

    model_names = ['Unet_Prior_needle_MSE_D8Lr0.0001_S1342',
                   'Unet_Prior_needle_DCT_MSE_D8Lr0.0001_S1342',
                   'Unet_Prior_needle_LoG0.1_MSE10.0_D8Lr0.0001_S1342',
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
            windowing(arr.transpose()[:, :, xy_pos], 0, 1)*255)
        cv2.imwrite(
            os.path.join(save_path, name+'_coronal.png'),
            windowing(arr.transpose()[:, xz_pos, :], 0, 1)*255)
        np.save(os.path.join(save_path, name+'_axial.npy'), arr.transpose()[:, :, xy_pos])
        np.save(os.path.join(save_path, name+'_coronal.npy'), arr.transpose()[:, xz_pos, :])

    test_ds = generate_testset_wo_prior(64, subject_idx, needle_idx, roll, center)
    model_names = ['Unet_woPrior_needle_MSE_D8Lr0.0001_S1342']
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
            windowing(arr.transpose()[:, :, xy_pos], 0, 1)*255)
        cv2.imwrite(
            os.path.join(save_path, name+'_coronal.png'),
            windowing(arr.transpose()[:, xz_pos, :], 0, 1)*255)
        np.save(os.path.join(save_path, name+'_axial.npy'), arr.transpose()[:, :, xy_pos])
        np.save(os.path.join(save_path, name+'_coronal.npy'), arr.transpose()[:, xz_pos, :])


def assemble_figure():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    pred_path = os.path.join('experiments', 'predictions')

    window_levels = {
        '0_0': (0.0, 1.0),
        '1_1': (.3689, .3915),
    }

    column_types = [
        'sparse',
        'gt',
        'Unet_Prior_needle_MSE_D8Lr0.0001_S1342',
        'Unet_Prior_needle_SSIM_D8Lr0.0001_S1342',
        'Unet_Prior_needle_MSSIM_D8Lr0.0001_S1342',
        'Unet_Prior_needle_LoG0.1_MSE10.0_D8Lr0.0001_S1342',
        'Unet_Prior_needle_DCT_MSE_D8Lr0.0001_S1342',
        'Unet_woPrior_needle_MSE_D8Lr0.0001_S1342',
    ]

    row_types = [
        ('0_0', 'axial'),
        ('0_0', 'coronal'),
        ('1_1', 'axial'),
        ('1_1', 'coronal'),
    ]

    fig, axs = plt.subplots(len(row_types), len(column_types)+1, figsize=(20, 8))

    def custom_psnr(pred, gt, max_val: float = 1.0):
        return 10*np.log10(max_val**2/np.mean((pred - gt)**2))

    for row_idx, row_item in enumerate(row_types):
        for col_idx, col_item in enumerate(column_types):
            img = np.load(os.path.join(pred_path, '1_1', f'{col_item}_{row_item[1]}.npy'))
            axs[row_idx, col_idx].imshow(
                img,
                vmin=window_levels[row_item[0]][0],
                vmax=window_levels[row_item[0]][1],
                cmap='gray')
            axs[row_idx, col_idx].axis('off')
            if col_idx != 1 and row_idx < 2:
                gt = np.load(os.path.join(pred_path, '1_1', f'gt_{row_item[1]}.npy'))
                psnr_val = custom_psnr(gt, img, max_val=1.)
                ssim_val = skssim(gt, img, data_range=1.)*100
                axs[row_idx, col_idx].text(0, 0, f'{psnr_val:.2f}dB, {ssim_val:.2f}\\%', va='top', color='yellow')

    axs[0, 0].set_title(r'$\mathbf{LQ}$')
    axs[0, 1].set_title(r'$\mathbf{HQ}$')
    axs[0, 2].set_title(r'$\mathbf{M}_{\mathbf{MSE}}$')
    axs[0, 3].set_title(r'$\mathbf{M}_{\mathbf{SSIM}}$')
    axs[0, 4].set_title(r'$\mathbf{M}_{\mathbf{MSSIM}}$')
    axs[0, 5].set_title(r'$\mathbf{M}_{\mathbf{LoG}}$')
    axs[0, 6].set_title(r'$\mathbf{M}_{\mathbf{DCT}}$')
    axs[0, 7].set_title(r'$\mathbf{M}_{\mathbf{no\_prior}}$')
    axs[0, 8].set_title(r'$\mathbf{SIRT}$')

    img = np.array(Image.open('/home/phernst/Documents/Memorial/isbi2022/sirt_N177c_Needle2_Pos2_12_wide_window_registered.png'))[..., 0]
    axs[0, 8].imshow(img, cmap='gray')
    axs[0, 8].text(0, 0, '23.90dB, 83.37\\%', va='top', color='yellow')

    img = np.array(Image.open('/home/phernst/Documents/Memorial/isbi2022/cor_N177c_Needle2_Pos2_12_wide_window_registered.png'))[..., 0]
    im1 = axs[1, 8].imshow(img, cmap='gray')
    axs[1, 8].text(0, 0, '20.84dB, 79.06\\%', va='top', color='yellow')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im1, cax=cbar_ax)
    # fig.subplots_adjust(right=0.8)

    img = np.array(Image.open('/home/phernst/Documents/Memorial/isbi2022/sirt_N177c_Needle2_Pos2_12_registered.png'))[..., 0]
    axs[2, 8].imshow(img, cmap='gray')

    img = np.array(Image.open('/home/phernst/Documents/Memorial/isbi2022/cor_N177c_Needle2_Pos2_12_narrow_window_registered.png'))[..., 0]
    im2 = axs[3, 8].imshow(img, cmap='gray')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im2, cax=cbar_ax)
    # fig.subplots_adjust(right=0.8)

    for i in range(len(row_types)):
        axs[i, 8].axis('off')

    axs[0, 0].set(ylabel='axial')
    axs[1, 0].set(ylabel='axial')
    axs[2, 0].set(ylabel='coronal')
    axs[3, 0].set(ylabel='coronal')

    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    plt.show()


def roi_figure_images():
    root_path = '/home/phernst/Documents/git/interventional-CT/experiments/predictions/1_1/'
    gt = np.load(os.path.join(root_path, 'gt_subject.npy')).astype(np.float32)
    noisy = np.load(os.path.join(root_path, 'sparse_subject.npy')).astype(np.float32)

    model_names = [
        'Unet_Prior_needle_MSE_D8Lr0.0001_S1342',
        'Unet_Prior_needle_MSSIM_D8Lr0.0001_S1342',
        'Unet_Prior_needle_LoG0.1_MSE10.0_D8Lr0.0001_S1342',
        'Unet_Prior_needle_SSIM_D8Lr0.0001_S1342',
        'Unet_Prior_needle_DCT_MSE_D8Lr0.0001_S1342',
        'Unet_woPrior_needle_MSE_D8Lr0.0001_S1342',
    ]

    mse = np.load(os.path.join(root_path, model_names[0]+'.npy')).astype(np.float32).transpose()
    mssim = np.load(os.path.join(root_path, model_names[1]+'.npy')).astype(np.float32).transpose()
    log = np.load(os.path.join(root_path, model_names[2]+'.npy')).astype(np.float32).transpose()
    ssim = np.load(os.path.join(root_path, model_names[3]+'.npy')).astype(np.float32).transpose()
    dct = np.load(os.path.join(root_path, model_names[4]+'.npy')).astype(np.float32).transpose()
    wo = np.load(os.path.join(root_path, model_names[5]+'.npy')).astype(np.float32).transpose()

    fig, ax = plt.subplots(2, 4, figsize=(8, 4))
    font = 20
    #270, 250
    img_num = 170
    #img_num = 2050
    im = ax[0, 0].imshow(noisy[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(right=0.8)
    ax[0, 1].imshow(gt[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)

    ax[0, 2].imshow(mse[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    # ax[0].text(0,-5,get_ssim(mse[img_num], gt[img_num]),fontsize=font)

    ax[0, 3].imshow(mssim[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    # ax[1].text(0,-5,get_ssim(mssim[img_num], gt[img_num]),fontsize=font)

    ax[1, 0].imshow(ssim[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    # ax[2].text(0,-5,get_ssim(ssim[img_num], gt[img_num]),fontsize=font)

    ax[1, 1].imshow(log[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    # ax[3].text(0,-5,get_ssim(log[img_num], gt[img_num]),fontsize=font)

    ax[1, 2].imshow(dct[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    # ax[4].text(0,-5,get_ssim(dct[img_num], gt[img_num]),fontsize=font)

    ax[1, 3].imshow(gt[:, img_num][20:120, 200:300],cmap='gray', vmin=0, vmax=1)
    # ax[5].text(0,-5,get_ssim(noisy[img_num], gt[img_num]),fontsize=font)
    for i in range(4):
        for j in range(2):
            ax[j, i].axis('off')
    # [20:120, 200:300]
    plt.show()


def unnormalize_window_values(window_value):
    return mu2hu(window_value * hu2mu(CARMH_GT_UPPER_99_PERCENTILE, .02), .02)


def calculate_all_window_values():
    print(f'narrow window [{unnormalize_window_values(.3689)}, {unnormalize_window_values(.3915)}]')
    print(f'wide window [{unnormalize_window_values(.0)}, {unnormalize_window_values(1.)}]')


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # generate_predictions(subject_idx=0, needle_idx=0, roll=0, center=(0, 0, -50), xy_pos=145, xz_pos=208)
    # generate_predictions(subject_idx=1, needle_idx=1, roll=180, center=(0, 0, 0), xy_pos=278, xz_pos=170)
    # extract_input_slices(subject_idx=0, needle_idx=0, roll=0, center=(0, 0, -50), xy_pos=145, xz_pos=208)
    # extract_input_slices(subject_idx=1, needle_idx=1, roll=180, center=(0, 0, 0), xy_pos=278, xz_pos=170)
    # assemble_figure()
    create_figure1_images()
    # roi_figure_images()
    # calculate_all_window_values()

# test[0], needle[0], center=(0, 0, -50), 146/469, xz 209, window: [.3689, .3915]
# test[1], needle[1], center=(0, 0, 0), 279/493, xz 170, roll 180

# sirt:
# axial: 253, [.2310, .2429] (narrow window), [0, .6452] (wide window)
# coronal: 211
