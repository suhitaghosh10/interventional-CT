import os
import tensorflow as tf
import random
import numpy as np
tfkl = tf.keras.layers
tfk = tf.keras
tfkb = tf.keras.backend
AUTOTUNE = tf.data.experimental.AUTOTUNE
from tensorflow.keras.callbacks import LearningRateScheduler as LRS
import traceback

SEED = 42