from utility.experiment_init import Experiment
expt = Experiment(seed=42)

from utility.common_imports import *
from utility.constants import *

import argparse

from dataset.head_carm.models.prior_unet import unet
from utility.utils import dct_blockwise_batch
from utility.weight_norm import AdamWithWeightnorm
from utility.logger_utils_prior import PlotReconstructionCallback
from dataset.head_carm.utility.constants import *
from tensorflow.keras.callbacks import LearningRateScheduler as LRS
from dataset.head_carm.utility.dataset_creation import generate_datasets_dct


def store_min_max(a, b, c, min, max):
    a_min = tf.math.reduce_min(a)
    b_min = tf.math.reduce_min(b)
    c_min = tf.math.reduce_min(c)
    temp = tf.minimum(tf.minimum(a_min, b_min), c_min)
    if temp < min:
        min = temp

    a_min = tf.math.reduce_max(a)
    b_min = tf.math.reduce_max(b)
    c_min = tf.math.reduce_max(c)
    temp = tf.maximum(tf.maximum(a_min, b_min), c_min)
    if temp > max:
        max = temp
    return min, max


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import tensorflow as tf

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    ds = generate_datasets_dct(2, 2)
    min = 1000
    max = -1000
    for data in ds:
        inp, gt, prior = data[...,0:1], data[...,1:2], data[...,2:3]
        inp_dct, gt_dct, prior_dct = dct_blockwise_batch(inp, threshold_val=0.012), \
                                     dct_blockwise_batch(gt, threshold_val=0.012),\
                                     dct_blockwise_batch(prior, threshold_val=0.012)
        min, max = store_min_max(inp_dct, gt_dct, prior_dct, min, max)
        del inp, gt, prior
        print(min, max)

