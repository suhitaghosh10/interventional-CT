DATA_PATH= '/project/sghosh/dataset/carm_head/'
#TRAIN_NUM = len(os.listdir(os.path.join(DATA_PATH, 'train'))) * 314 * 64
# TRAIN_NUM = 32 * 314 * 64 # no of patients = 32
# VAL_NUM = 2 * 314
# TEST_NUM = 2 * 314
AUG_NUM = 1
TRAIN_NUM = 17180 * AUG_NUM # no of patients = 32
VAL_NUM = 3877
TEST_NUM = 352  # TODO

TB_PATH = '/scratch/sghosh/VAE/logdir'

NR_CHANNELS=1
DATASET_NAME = 'carmhead/image_2D'

#CODEBOOK_SIZE=512 #K
CODEBOOK_SIZE=512 #K
HIDDEN_NUM = 128 #hidden units in conv/convT
XY_DIM =16 #xy dim of image
LR = 1e-4

IMG_DIM_ORIG_2D = (512, 512, 1)
IMG_DIM_INP_2D = (384, 384, 1)
# Load dataset
IMGS_2D_SHARDS_PATH = '/project/sghosh/dataset/carm_head/tf_shards/'
IMGS_2D_DCT_SHARDS_PATH = '/project/sghosh/dataset/carm_head/dct/tf_shards/'
HEAD_PROJECTIONS: str = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/cone-beam/'
NEEDLE_PROJECTIONS: str = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/needles/'
VALIDATION_RECORDS: str = '/project/sghosh/dataset/mayoclinic/Head/validation/'
TEST_RECORDS: str = '/project/sghosh/dataset/mayoclinic/Head/test/'

CARMH_IMG_LOW_5_PERCENTILE = -1000.0
CARMH_IMG_UPPER_99_PERCENTILE = 1406.5560302734375
CARMH_GT_LOW_5_PERCENTILE = -1253.4764404296875
CARMH_GT_UPPER_99_PERCENTILE = 1720.43359375
#CARM_DCT_MIN = -73.75
#CARM_DCT_MAX = 169.0
CARM_DCT_MIN = -3.691
CARM_DCT_MAX = 8.01


##### NEVER EVER DEFINE ANY TENSORFLOW VARIABLE HERE!!!!!!