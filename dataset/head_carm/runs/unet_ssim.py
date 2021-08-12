from utility.experiment_init import Experiment
expt = Experiment(seed=42)

from utility.common_imports import *
from utility.constants import *
import argparse

from dataset.head_carm.dataset_creation import generate_perceptual_dataset
from dataset.head_carm.models import unet
from utility.utils import ssim, psnr, mse, lr_scheduler_linear, multiscale_ssim_l2, mssim
from utility.weight_norm import AdamWithWeightnorm
from utility.logger_utils_prior import PlotReconstructionCallback
from dataset.head_carm.constants import *
from tensorflow.keras.callbacks import LearningRateScheduler as LRS

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=2000, help='Number of training epochs')
parser.add_argument('-bs', '--batch', type=int, default=64, help='Batch size for training')
parser.add_argument('-bf', '--buffer', type=int, default=512, help='Buffer size for shuffling')
parser.add_argument('-d', '--d', type=int, default=8, help='starting embeddding dim')  # 128
parser.add_argument('-g', '--gpu', type=str, default='0, 1', help='gpu num')
parser.add_argument('-l', '--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-eager', '--eager', type=bool, default=False, help='eager mode')
parser.add_argument('-path', '--path', type=str, default='/project/sghosh/experiments/', help='path to experiments folder')

args = parser.parse_args()
gpu_id = args.gpu
nb_gpus = len(gpu_id.split(','))
bs = args.batch * nb_gpus

assert np.mod(bs, nb_gpus) == 0, \
    'batch_size should be a multiple of the nr. of gpus. ' + \
    'Got batch_size %d, %d gpus' % (bs, nb_gpus)
# Training parameters
epochs = args.epochs
buffer = args.buffer  # for shuffling
d = args.d
lr = args.lr
augm_no = 1
save_by = 'val_masked_ssim'
show_summary = True
is_eager = args.eager
scratch_dir = args.path
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

ds, vds, teds = generate_perceptual_dataset(IMGS_2D_SHARDS_PATH, bs, buffer)
steps = (TRAIN_NUM * augm_no) // bs

NAME = 'Unet_Prior_needlev2_SSIM_MSE_priorangle15'+ '_D' + str(d) + 'Lr' + str(lr)+ '_d'
CHKPNT_PATH = scratch_dir+'carmh/UnetPrior_needlev2_SSIM_MSE__priorangle15_seed'+str(expt.get_seed())+'/chkpnt/'
os.makedirs(CHKPNT_PATH, exist_ok=True)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a s<trategy scope.
with strategy.scope():
    act = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Define model
    unet_cls = unet()
    model = unet_cls.build_model(d=d, act=act)
    decay_steps = 1000
    coslr = tf.keras.experimental.CosineDecayRestarts(
        lr, decay_steps, t_mul=(steps * epochs) // decay_steps, m_mul=1.0, alpha=lr // 10
    )
    optimizer = AdamWithWeightnorm(learning_rate=coslr)
   # optimizer = AdamWithWeightnorm(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  run_eagerly=is_eager,
                  loss=multiscale_ssim_l2(IMG_DIM_INP_2D, mse_weight=1., ssim_weight=1.0),
                  metrics=[mse(IMG_DIM_INP_2D),
                           ssim(IMG_DIM_INP_2D),
                           mssim(IMG_DIM_INP_2D),
                           psnr(IMG_DIM_INP_2D)
                           ])

    model.summary()
    model.run_eagerly = is_eager  # set true if debug on

    # # Callbacks
    TENSORBOARD_PATH = os.path.join('/scratch/sghosh/VAE/logdir', 'Unet_Prior_'+DATASET_NAME, NAME)
    tensorboard_clbk = tfk.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH)

    plot_clbk = PlotReconstructionCallback(logdir=TENSORBOARD_PATH,
                                           test_ds=teds,
                                           chkpoint_path=CHKPNT_PATH,
                                           save_by=save_by,
                                           save_by_decrease=False,
                                           log_on_epoch_end=True,
                                           step_num=1000
                                           )
    chkpnt_cb = tfk.callbacks.ModelCheckpoint(os.path.join(CHKPNT_PATH, CHKPOINT_NAME),
                                              monitor=save_by,
                                              verbose=1,
                                              save_freq='epoch',
                                              mode='max',
                                              save_best_only=True,
                                              save_weights_only=True)
    es = tfk.callbacks.EarlyStopping(monitor=save_by, mode='max', verbose=1, patience=50, min_delta=1e-4)
    LRDecay = tfk.callbacks.ReduceLROnPlateau(monitor=save_by, factor=0.5, patience=5, verbose=1, mode='max',
                                              min_lr=1e-8,
                                              min_delta=0.01)

    lrs = LRS(lr_scheduler_linear, verbose=1)
    callbacks = [tensorboard_clbk, plot_clbk, chkpnt_cb]

    try:
        ckpt = tf.train.Checkpoint( net=model, optimizer=optimizer)
        ckpt.restore(os.path.join(CHKPNT_PATH , CHKPOINT_NAME))
        print("Restored from {}".format(CHKPNT_PATH))
    except:
        print("Initializing from scratch.")

    # # Fit
    #model.summary()
    model.fit(ds, validation_data=vds, epochs=epochs, callbacks=callbacks,
              steps_per_epoch=steps, validation_steps=VAL_NUM // bs)
