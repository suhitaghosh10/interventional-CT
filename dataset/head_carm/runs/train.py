import argparse
import os
from os.path import join as pjoin

import tensorflow as tf
from tensorflow import errors as tfe
from tensorflow import keras as tfk
import torch

from dataset.head_carm.models.prior_unet import UNet
from dataset.head_carm.utility.constants import TRAIN_NUM, AUG_NUM, VAL_NUM, \
    IMG_DIM_INP_2D, VALIDATION_RECORDS_13_PATH, TEST_RECORDS_13_PATH
from dataset.head_carm.utility.dataset_creation import generate_datasets
from dataset.head_carm.utility.utils import apply_tf_gpu_memory_growth, \
    generate_loss_dict, set_available_gpus, run_distributed, ClearMemory
from one_cycle_lr import OneCycleLr
from utility.constants import SPARSE_PROJECTION_NUM, CHKPOINT_NAME
from utility.experiment_init import Experiment
from utility.utils import ssim, psnr, mse, mssim
from utility.weight_norm import AdamWithWeightnorm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-loss', '--loss', type=str, default='MSSIM_MSE', help='loss type', choices=generate_loss_dict())
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-bs', '--batch', type=int, default=32, help='Batch size for training')
    parser.add_argument('-bf', '--buffer', type=int, default=2, help='Buffer size for shuffling')
    parser.add_argument('-d', '--d', type=int, default=8, help='starting embeddding dim')  # 128
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu num')
    parser.add_argument('-l', '--lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument('-eager', '--eager', type=bool, default=False, help='eager mode')
    parser.add_argument('-path', '--path', type=str, default='experiments', help='path to experiments folder')
    return parser.parse_args()


def run_training(cmd_args, nb_gpus):
    loss_dict = generate_loss_dict()
    assert cmd_args.loss in loss_dict

    # Training parameters
    batch_size = cmd_args.batch * nb_gpus
    model_loss = cmd_args.loss
    loss_fn = loss_dict[model_loss]
    save_by = 'val_masked_psnr'

    tds, vds, _ = generate_datasets(
        VALIDATION_RECORDS_13_PATH,
        TEST_RECORDS_13_PATH,
        batch_size,
        cmd_args.buffer,
    )
    steps = (TRAIN_NUM * AUG_NUM) // batch_size

    model_name = f'Unet_Prior_needle_{model_loss}_' \
        f'D{cmd_args.d}Lr{cmd_args.lr}_S{SPARSE_PROJECTION_NUM}'
    chkpnt_path = pjoin(
        cmd_args.path,
        f'{model_name}_{Experiment.get_seed()}',
        'chkpnt')
    os.makedirs(chkpnt_path, exist_ok=True)

    # Define model
    model = UNet.build_model(d=cmd_args.d, act=tfk.layers.LeakyReLU(alpha=0.2))

    onecycle_lr = OneCycleLr(
        max_lr=cmd_args.lr,
        total_steps=steps*cmd_args.epochs,
    )
    optimizer = AdamWithWeightnorm(learning_rate=onecycle_lr)

    model.compile(
        optimizer=optimizer,
        run_eagerly=cmd_args.eager,
        loss=loss_fn,
        metrics=[
            mse(IMG_DIM_INP_2D),
            mssim(IMG_DIM_INP_2D),
            ssim(IMG_DIM_INP_2D),
            psnr(IMG_DIM_INP_2D),
        ],
    )
    model.summary()
    model.run_eagerly = cmd_args.eager  # set true if debug on

    # Callbacks
    tensorboard_path = 'logs'
    tensorboard_cb = tfk.callbacks.TensorBoard(log_dir=tensorboard_path)

    chkpnt_cb = tfk.callbacks.ModelCheckpoint(
        pjoin(chkpnt_path, CHKPOINT_NAME),
        monitor=save_by,
        verbose=1,
        save_freq='epoch',
        mode='max',
        save_best_only=True,
        # save_weights_only=True,
    )
    clear_cb = ClearMemory()

    callbacks = [tensorboard_cb, chkpnt_cb, clear_cb]

    try:
        ckpt = tf.train.Checkpoint(net=model, optimizer=optimizer)
        ckpt.restore(pjoin(chkpnt_path, CHKPOINT_NAME))
        print(f"Restored from {chkpnt_path}")
    except tfe.NotFoundError:
        print("Initializing from scratch.")

    # Fit
    model.fit(
        tds,
        validation_data=vds,
        epochs=cmd_args.epochs,
        callbacks=callbacks,
        steps_per_epoch=steps,
        validation_steps=VAL_NUM // batch_size,
    )


def main(cmd_args):
    Experiment.set_seed(42)

    nb_gpus = set_available_gpus(cmd_args)
    apply_tf_gpu_memory_growth()

    print('Torch cuda available:', torch.cuda.is_available())

    if nb_gpus > 1:
        run_distributed(run_training, cmd_args, nb_gpus)
    else:
        run_training(cmd_args, nb_gpus)


if __name__ == '__main__':
    main(parse_arguments())
