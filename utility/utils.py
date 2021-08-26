from utility.common_imports import *
from utility.constants import *
import tensorflow_addons as tfa
#from utility.tensorflow_imported.filter_ops import gaussian, laplacian
AUGMENTATION_MAX_ANGLE = 1.0 #radian
GAUSSIAN_SIGMA = 1.
GAUSSIAN_LAPLACIAN_K_SIZE=7

def ssim(dim, paddings=None):
    def masked_ssim(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        mssim = tf.image.ssim(y_true, y_pred, max_val=1.)
        return mssim

    return masked_ssim

def psnr(dim, paddings=None):
        def masked_psnr(y_true, y_pred):
            if paddings is not None:
                temp = tf.constant(value=1.0, shape=dim)
                mask = tf.pad(temp, paddings=paddings)
                y_true = mask * y_true
                y_pred = mask * y_pred
            psnr = tf.image.psnr(y_true, y_pred, max_val=1.)
            return psnr

        return masked_psnr

def mse(dim, paddings=None, weight=1):
        def masked_mse(y_true, y_pred):
            if paddings is not None:
                temp = tf.constant(value=1.0, shape=dim)
                mask = tf.pad(temp, paddings=paddings)
                y_true = mask * y_true
                y_pred = mask * y_pred
            mse_val = tf.metrics.mean_squared_error(y_pred, y_true)
            return weight * mse_val

        return masked_mse


@tf.function
def augment_prior(image:tf.Tensor, annotation:tf.Tensor):
        MAX_ANGLE_PRIOR = tf.constant(0.25)
        #MAX_ANGLE_PRIOR= 0.0873
        NEEDLE_MAX_ANGLE = tf.constant(3.14)
        image = tf.cast(image, dtype=tf.float32)
        annotation = tf.cast(annotation, dtype=tf.float32)
        needle_type = tf.random.uniform(minval=0, maxval=2, dtype=tf.int32, shape=[])
        #needle_type = tf.constant(1)
        needle = _get_needle_tensor(needle_type)
        #needle = _get_needle_tensor(tf.constant(1, dtype=tf.int32))
        # 0-none 1-rotate 2-scale 3-flip 4-flip_rotate
        rand_num = tf.random.uniform(shape=[], minval=0, maxval=5, dtype=tf.int32, seed=10)
        #rand_num = 4
        angle = tf.random.uniform(shape=[], minval=-AUGMENTATION_MAX_ANGLE, maxval=AUGMENTATION_MAX_ANGLE,
                                           dtype=tf.float32, seed=10)
        needle_angle = tf.random.uniform(shape=[], minval=-NEEDLE_MAX_ANGLE, maxval=NEEDLE_MAX_ANGLE,
                                  dtype=tf.float32, seed=15)
        # dont rotate the prior image too much!
        prior_angle = tf.random.uniform(shape=[], minval=-MAX_ANGLE_PRIOR, maxval=MAX_ANGLE_PRIOR,
                                  dtype=tf.float32, seed=10)
        scale_ratio = tf.random.uniform(shape=[], minval=0.8, maxval=1.0, dtype=tf.float32, seed=10)

        if tf.equal(rand_num, tf.constant(1)):
            image, annotation = _rotate(image, angle), _rotate(annotation, angle)
        if tf.equal(rand_num, tf.constant(2)):
            image, annotation = _scale(image, scale_ratio), _scale(annotation, scale_ratio)
        if tf.equal(rand_num, tf.constant(3)):
            image, annotation = _flip(image), _flip(annotation)
        if tf.equal(rand_num, tf.constant(4)):
            image, annotation = _flip_rotate(image, angle), _flip_rotate(annotation, angle)

        if tf.equal(needle_type, tf.constant(0)):
            needle = _rotate(needle, needle_angle)
            image = tf.where(needle > 0., tf.random.uniform(shape=[], minval=0.9, maxval=1.5, dtype=tf.float32), image)
        else:
            needle = _translate(needle)
            image = tf.where(needle > 0., tf.random.uniform(shape=[], minval=1.0, maxval=1.5, dtype=tf.float32),
                                     image)

        prior = _rotate(annotation, prior_angle)
        annotation = tf.where(needle > 0., 1., annotation)
        return [image, prior], annotation

def _get_needle_numpy(needle_type):
    #circle_flag = np.random.randint(0, 1)
    needle_type
    image_shape = (384, 384, 1)
    x = image_shape[0]
    y = image_shape[1]
    max_x = x // 2
    max_y = y // 2
    start_x = np.random.randint(low=x // 3, high=max_x)
    start_y = np.random.randint(low=y // 3, high=max_y)
    needl = np.zeros(shape=image_shape)
    if needle_type == 0: # rectangle
        min_l = 10
        max_l = np.maximum(x, y)
        length = np.random.randint(low=min_l, high=max_l)
        width = np.random.randint(low=1, high=3)
        needl[start_x:start_x + width, start_y:start_y + length, :] = 1.0

    if needle_type == 1: # circle
        min_l = 2
        max_l = 5
        length = np.random.randint(low=min_l, high=max_l)
        needl[start_x:start_x + length, start_y:start_y + length, :] = 1.0

    needl[needl > 1] = 1.0
    needl[needl < 1] = 0.0

    return needl


@tf.function(input_signature=[tf.TensorSpec(None, tf.int32)])
def _get_needle_tensor(needle_type: tf.Tensor):
    needle_tensor = tf.numpy_function(_get_needle_numpy, [needle_type], tf.double)
    return needle_tensor

@tf.function
def insert_needle(image: tf.Tensor, annotation: tf.Tensor):
        image = tf.cast(image, dtype=tf.float32)
        annotation = tf.cast(annotation, dtype=tf.float32)
        prior = tf.cast(annotation, dtype=tf.float32)
        #needle = _get_needle_tensor(tf.random.uniform(minval=0, maxval=2, dtype=tf.int32))
        needle = _get_needle_tensor(tf.constant(0, dtype=tf.int32))
        image = tf.where(needle > 0., tf.random.uniform(shape=[], minval=0.9,maxval=1.5, dtype=tf.float32), image)
        annotation = tf.where(needle > 0, 1.0, annotation)
        return [image, prior], annotation


@tf.function
def _rotate(image: tf.Tensor, angle: tf.Tensor):
        return tfa.image.rotate(image, angle, interpolation='BILINEAR')

@tf.function
def _translate(image: tf.Tensor):
        x = tf.random.uniform(minval=-50, maxval=50, dtype=tf.int32, shape=[])
        y = tf.random.uniform(minval=-50, maxval=50, dtype=tf.int32, shape=[])
        return tfa.image.translate(image, [x,y], interpolation='nearest', fill_mode= 'constant', fill_value = 0.0)

@tf.function
def _flip(image: tf.Tensor):
        return tf.image.flip_left_right(image)

@tf.function
def _flip_rotate(image: tf.Tensor, angle: tf.Tensor):
        image = tf.image.flip_left_right(image)
        return tfa.image.rotate(image, angle, interpolation='BILINEAR')

@tf.function
def _scale(image: tf.Tensor, ratio: tf.Tensor, dim=[384, 384, 1]):
        print(ratio)
        x1 = y1 = 0.5 - (0.5 * ratio)
        x2 = y2 = 0.5 + (0.5 * ratio)
        boxes = [x1,y1,x2,y2]
        boxes = tf.cast(tf.reshape(boxes, shape=(1,4)), dtype=tf.float32)
        scaled = tf.image.crop_and_resize([image], boxes=boxes, box_indices=tf.zeros(1, dtype=tf.int32),
                                         crop_size=(dim[0], dim[1]))
        scaled = tf.reshape(scaled, shape=(dim[0], dim[1], dim[2]))
        #return tf.cast(scaled, dtype=tf.float16)
        return scaled


def lr_scheduler_linear(epoch, lr):
    decay_rate = 0.5
    decay_step = 5
    print(epoch)
    if (epoch > 0) and (epoch % decay_step == 0):
        return lr * decay_rate
    return lr

def multiscale_ssim_l2(dim, weight=1, paddings=None, alpha=0.84):
        def masked_mssim_l1(y_true, y_pred):
            if paddings is not None:
                temp = tf.constant(value=1.0, shape=dim)
                mask = tf.pad(temp, paddings=paddings)
                y_true = mask * y_true
                y_pred = mask * y_pred
            ms_ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.)
            l1 = tf.reduce_mean(tf.keras.metrics.MAE(y_true, y_pred), axis=(-1, -2))
            loss = (alpha * ms_ssim) + ((1 - alpha) * l1)
            return weight * loss

        return masked_mssim_l1


def mssim(dim, weight=1., paddings=None):
    def masked_mssim(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        ms_ssim = weight * tf.image.ssim_multiscale(y_true, y_pred, max_val=1.)
        return ms_ssim
    return masked_mssim
