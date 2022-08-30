
import tensorflow as tf
import os
from dataset.head_carm.models.prior_unet import unet
from utility.weight_norm import AdamWithWeightnorm
from utility.utils import ssim, psnr, mse, mssim, multiscale_ssim_l2, multiscale_ssim_l1
from utility.constants import *
from dataset.head_carm.utility.constants import *
from dataset.head_carm.utility.dataset_creation import generate_test_dataset
from utility.experiment_init import *

d = 8
lr = 1e-4

def get_model_ssim(path):
    # Define model
    unet_cls = unet()

    model = unet_cls.build_model(d=d, act=act)
    decay_steps = 1000
    optimizer = AdamWithWeightnorm(learning_rate=1e-4)
    # optimizer = AdamWithWeightnorm(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  run_eagerly=False,
                  loss=mse(IMG_DIM_INP_2D),
                  metrics=[ssim(IMG_DIM_INP_2D),
                           psnr(IMG_DIM_INP_2D),
                           mse(IMG_DIM_INP_2D),
                           mssim(IMG_DIM_INP_2D)
                           ])
    CHKPNT_PATH = os.path.join(path)
    model.load_weights(os.path.join(CHKPNT_PATH, CHKPOINT_NAME) )
    return model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    expt = Experiment()
    act = tf.keras.layers.LeakyReLU(alpha=0.2)


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    save_path = 'tmp'
    #os.mkdir(save_path)
    test_ds = generate_test_dataset('/mnt/nvme2/mayoclinic/ds_test/13/')
    test_ds = test_ds.batch(1)
    # gt = np.zeros((10296, 384,384), dtype=np.float32)
    # i=0
    # for data in test_ds:
    #     gt[i] = data[1][0,...,0]
    #     i+= 1
    # np.save('/project/sghosh/experiments/predictions/13/gt.npy', gt)
    model_names = [#'Unet_Prior_needle_MSE_D8Lr0.0001_S1342',
                   #'Unet_Prior_needle_DCT_MSE_D8Lr0.0001_S1342',
                   #'Unet_Prior_needle_LoG0.1_MSE10.0_D8Lr0.0001_S1342',
                #    'Unet_woPrior_needle_MSE_D8Lr0.0001_S1342'
                  'Unet_Prior_needle_MSSIM_D8Lr0.0001_S1342',
   # 'Unet_Prior_needle_SSIM_D8Lr0.0001_S1342'
    ]
    import nibabel as nib
    import numpy as np
    for name in model_names:
        print(name)
        model = get_model_ssim(os.path.join('/home/phernst/Documents/git/interventional-CT/tmp_experiments', name, 'chkpnt'))
        results = model.evaluate(test_ds, batch_size=128)
        print('ssim psnr mse mssim', format(results[1], ".5f"), format(results[2], ".5f"), format(results[3], ".3e"), format(results[3], ".5f"))
    
        arr = model.predict(test_ds, batch_size=128)
        np.save(os.path.join(save_path, name+'.npy'), arr)
    img = arr[:512, ..., 0]
    img = np.transpose(img, (2, 1, 0))
    img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, os.path.join(save_path, name+'.nii.gz'))