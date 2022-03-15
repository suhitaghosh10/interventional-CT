import os
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset.head_carm.models.prior_unet import UNet
from dataset.head_carm.utility.constants import IMG_DIM_INP_2D,\
    TEST_RECORDS_13_PATH, VALIDATION_RECORDS_13_PATH
from dataset.head_carm.utility.dataset_creation import generate_datasets
from dataset.head_carm.utility.utils import generate_loss_dict
from utility.utils import ssim, psnr, mse


def main(loss_type: str):
    assert loss_type in generate_loss_dict()
    prediction_dir: str = f'predictions_{loss_type}'
    os.makedirs(prediction_dir, exist_ok=True)

    start_filter_size = 8
    act = tf.keras.layers.LeakyReLU(alpha=0.2)
    model = UNet.build_model(d=start_filter_size, act=act)
    _ = model.load_weights(pjoin(
        'experiments',
        f'Unet_Prior_needle_{loss_type}_D8Lr0.0006_S13_42',
        'chkpnt',
        'cp.ckpt',
    )).expect_partial()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=mse(IMG_DIM_INP_2D),
        metrics=[
            mse(IMG_DIM_INP_2D),
            ssim(IMG_DIM_INP_2D),
            psnr(IMG_DIM_INP_2D),
        ]
    )

    tds = generate_datasets(
        VALIDATION_RECORDS_13_PATH,
        TEST_RECORDS_13_PATH,
        1,
        2,
    )[2].batch(batch_size=64)
    tds_iter = iter(tds)
    for idx, batch in tqdm(enumerate(tds_iter)):
        prediction = model(batch[0], training=False)[..., 0].numpy()
        img = nib.Nifti1Image(prediction.transpose(), np.eye(4))
        nib.save(img, pjoin(prediction_dir, f'pred_{idx}.nii.gz'))
        img = nib.Nifti1Image(batch[0][:, 0, ..., 0].numpy().transpose(), np.eye(4))
        nib.save(img, pjoin(prediction_dir, f'sparse_{idx}.nii.gz'))
        img = nib.Nifti1Image(batch[1][..., 0].numpy().transpose(), np.eye(4))
        nib.save(img, pjoin(prediction_dir, f'full_{idx}.nii.gz'))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    main(loss_type="SSIM_MSE")
