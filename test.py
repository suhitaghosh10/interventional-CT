from os.path import join as pjoin

import nibabel as nib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset.head_carm.models.prior_unet import unet as UNet
from dataset.head_carm.utility.dataset_creation import generate_datasets
from utility.utils import ssim, psnr, mse


def main():
    prediction_dir: str = 'predictions'

    start_filter_size = 8
    act = tf.keras.layers.LeakyReLU(alpha=0.2)
    model = UNet().build_model(d=start_filter_size, act=act)
    _ = model.load_weights(pjoin('checkpoints', 'cp.ckpt')).expect_partial()

    img_dim_inp_2d = (384, 384, 1)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=mse(img_dim_inp_2d, weight=1.),
                  metrics=[mse(img_dim_inp_2d),
                           ssim(img_dim_inp_2d),
                           psnr(img_dim_inp_2d)
                           ])

    tds = generate_datasets()[2].batch(batch_size=64)
    tds_iter = iter(tds)
    for idx, batch in tqdm(enumerate(tds_iter)):
        prediction = model(batch[0])[..., 0].numpy()
        img = nib.Nifti1Image(prediction.transpose(), np.eye(4))
        nib.save(img, pjoin(prediction_dir, f'pred_{idx}.nii.gz'))
        img = nib.Nifti1Image(batch[0][:, 0, ..., 0].numpy().transpose(), np.eye(4))
        nib.save(img, pjoin(prediction_dir, f'sparse_{idx}.nii.gz'))
        img = nib.Nifti1Image(batch[1][..., 0].numpy().transpose(), np.eye(4))
        nib.save(img, pjoin(prediction_dir, f'full_{idx}.nii.gz'))


if __name__ == '__main__':
    main()
