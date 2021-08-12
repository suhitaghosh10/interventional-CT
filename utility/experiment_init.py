from utility.common_imports import *
class Experiment():
    def __init__(self, seed=456):
        self.seed = seed
        # Set a seed value
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED']=str(seed)
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed)
        # 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed)
        # 4. Set `tensorflow` pseudo-random generator at a fixed value
        tf.random.set_seed(seed)

       # os.environ['TF_DETERMINISTIC_OPS'] = '1'
       # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        #tf.config.threading.set_inter_op_parallelism_threads(1)
        #tf.config.threading.set_intra_op_parallelism_threads(1)

    def get_seed(self):
        return str(self.seed)