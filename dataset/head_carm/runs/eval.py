
import csv
from os.path import join as pjoin

import numpy as np
import tensorflow as tf

from dataset.head_carm.models.prior_unet import UNet
from dataset.head_carm.utility.constants import IMG_DIM_INP_2D, \
    TEST_RECORDS_13_PATH
from dataset.head_carm.utility.dataset_creation import generate_test_dataset
from utility.constants import CHKPOINT_NAME
from utility.experiment_init import Experiment
from utility.utils import ssim, psnr, mse, mssim
from utility.weight_norm import AdamWithWeightnorm


def get_model(path):
    # Define model
    model = UNet.build_model(d=8, act=tf.keras.layers.LeakyReLU(alpha=0.2))
    optimizer = AdamWithWeightnorm(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
        loss=mse(IMG_DIM_INP_2D),
        metrics=[
            ssim(IMG_DIM_INP_2D),
            psnr(IMG_DIM_INP_2D),
            mse(IMG_DIM_INP_2D),
            mssim(IMG_DIM_INP_2D)
        ])
    chkpnt_path = pjoin(path)
    _ = model.load_weights(pjoin(chkpnt_path, CHKPOINT_NAME)).expect_partial()
    return model


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    Experiment.set_seed(42)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    save_path = '/project/sghosh/experiments/predictions/13/'
    # os.mkdir(save_path)
    test_ds = generate_test_dataset(TEST_RECORDS_13_PATH)
    test_ds = test_ds.batch(1)
    # gt = np.zeros((10296, 384,384), dtype=np.float32)
    # i=0
    # for data in test_ds:
    #     gt[i] = data[1][0,...,0]
    #     i+= 1
    # np.save('/project/sghosh/experiments/predictions/13/gt.npy', gt)
    model_names = {
        'DCT_MSE': 'Unet_Prior_needle_DCT_MSE_D8Lr0.0006_S13_42',
        'L1': 'Unet_Prior_needle_L1_D8Lr0.0006_S13_42',
        'MSE': 'Unet_Prior_needle_MSE_D8Lr0.0006_S13_42',
        'MSSIM_MSE': 'Unet_Prior_needle_MSSIM_MSE_D8Lr0.0006_S13_42',
        'MSSIM_WMSE': 'Unet_Prior_needle_MSSIM_WMSE_D8Lr0.0006_S13_42',
        'SSIM_MSE': 'Unet_Prior_needle_SSIM_MSE_D8Lr0.0006_S13_42',
        # 'Unet_Prior_needle_LoG_MSE_D8Lr0.0006_S13_42',
    }

    metrics = {loss: [] for loss in model_names}
    for loss, name in model_names.items():
        print(loss)
        model = get_model(pjoin('/home/phernst/Documents/git/interventional-CT/experiments', name, 'chkpnt'))
        results = model.evaluate(test_ds, batch_size=128)
        # print('ssim psnr mse mssim')
        # print(f'{results[1]:.5f}, {results[2]:.5f}, {results[3]:.3e}, '
        #       f'{results[4]:.5f}')
        metrics[loss] += results[1:]

        # arr = model.predict(test_ds, batch_size=128)
        # np.save(pjoin(save_path, f'{name}.npy'), arr)

    with open("metrics.csv", 'w', encoding='utf-8') as fhandle:
        csvwriter = csv.writer(fhandle)
        csvwriter.writerow(['loss', 'ssim', 'psnr', 'mse', 'mssim'])
        for loss, values in metrics.items():
            csvwriter.writerow([loss] + [f'{m}' for m in values])


if __name__ == '__main__':
    main()
