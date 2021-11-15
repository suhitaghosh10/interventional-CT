DATA_PATH= '/project/sghosh/dataset/carm_head/'
AUG_NUM = 2
TRAIN_NUM = 17180
VAL_NUM = 11631
TEST_NUM = 10296

TB_PATH = '/scratch/sghosh/VAE/logdir'

NR_CHANNELS=1
DATASET_NAME = 'carmhead/image_2D'

HIDDEN_NUM: int = 128 #hidden units in conv/convT
XY_DIM: int =16 #xy dim of image
LR = 1e-4

IMG_DIM_ORIG_2D = (512, 512, 1)
IMG_DIM_INP_2D = (384, 384, 1)
# Load dataset
#IMGS_2D_SHARDS_PATH = '/project/sghosh/dataset/carm_head/tf_shards/'
#IMGS_2D_DCT_SHARDS_PATH = '/project/sghosh/dataset/carm_head/dct/tf_shards/'


TRAIN_HELICAL_PRIOR_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/validation/priors/'
TRAIN_CONEBEAM_PROJECTIONS_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/cone-beam/'
NEEDLE_PROJECTIONS_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/high_dose_projections/needles/'
VALIDATION_RECORDS_18_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/validation/'
TEST_RECORDS_18_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/test/'
VALIDATION_RECORDS_2_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/validation/2/'
TEST_RECORDS_2_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/test/2/'
VALIDATION_RECORDS_13_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/validation/13/'
TEST_RECORDS_13_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/test/13/'
VALIDATION_RECORDS_15_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/validation/15/'
TEST_RECORDS_15_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/test/15/'
VALIDATION_RECORDS_4_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/validation/4/'
TEST_RECORDS_4_PATH: str = '/project/sghosh/dataset/mayoclinic/Head/test/4/'
#JSON_PATH:str = '../utility/train_valid_test.json'
JSON_PATH:str = 'dataset/head_carm/utility/train_valid_test.json'

CARMH_IMG_LOW_5_PERCENTILE: float = -1000.0
CARMH_IMG_UPPER_99_PERCENTILE: float = 1406.5560302734375
CARMH_GT_LOW_5_PERCENTILE: float = -1253.4764404296875
CARMH_GT_UPPER_99_PERCENTILE: float = 1720.43359375
#CARM_DCT_MIN = -73.75
#CARM_DCT_MAX = 169.0
CARM_DCT_MIN: float = -25998.479
CARM_DCT_MAX: float = 43110.605


##### NEVER EVER DEFINE ANY TENSORFLOW VARIABLE HERE!!!!!!