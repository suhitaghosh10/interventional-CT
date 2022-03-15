import os
import random

import numpy as np
import tensorflow as tf


class Experiment:
    @classmethod
    def set_seed(cls, seed=42):
        cls.seed = seed
        # Set a seed value
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        os.environ['PYTHONHASHSEED'] = str(seed)
        # 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed)
        # 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed)
        # 4. Set `tensorflow` pseudo-random generator at a fixed value
        tf.random.set_seed(seed)

        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)

    @classmethod
    def get_seed(cls):
        return str(cls.seed)
