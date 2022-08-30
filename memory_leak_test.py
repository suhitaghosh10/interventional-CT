import numpy as np
import tensorflow as tf

dims = 1024
seq_len = 1024

use_gpu = True

tf.profiler.experimental.start(
    logdir="test_fft_convolution_speed_log" + ("" if use_gpu else "_cpu")
)

def my_function():
    u = tf.zeros((1, 1, 480, 360, seq_len))
    v = tf.ones((1, 1, 480, 360, seq_len))

    fu = tf.signal.rfft(u)
    fv = tf.signal.rfft(v)

    fw = fu * fv
    w = tf.signal.irfft(fw, name="iw")[..., :seq_len]
    return w

with tf.device("/GPU:0" if use_gpu else "/CPU:0"):
    w = my_function()
    print(w.numpy().shape)