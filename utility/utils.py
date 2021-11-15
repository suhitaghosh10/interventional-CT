from utility.common_imports import *
from utility.constants import *
import tensorflow_addons as tfa
from utility.tensorflow_imported.filter_ops import gaussian, laplacian
AUGMENTATION_MAX_ANGLE = 1.0 #radian
GAUSSIAN_SIGMA = 1.
GAUSSIAN_LAPLACIAN_K_SIZE=7

a = [1, 2, 3]
iter_a = iter(a)

def ssim(dim, paddings=None, apply_idct=False, dct_min=None, dct_max=None):
    def masked_ssim(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        if apply_idct:
            min_tf = tf.constant(dct_min, dtype=tf.float16)
            max_tf = tf.constant(dct_max, dtype=tf.float16)
            dct_diff = (max_tf - min_tf)
            y_true = tf.cast((tf.cast(y_true, dtype=tf.float16) * dct_diff) + min_tf, dtype=tf.float32)
            y_pred = tf.cast((tf.cast(y_pred, dtype=tf.float16) * dct_diff) + min_tf, dtype=tf.float32)
            ssim = tf.image.ssim(image_batch_idct(y_true), image_batch_idct(y_pred), max_val=1.)
        else:
            ssim = tf.image.ssim(y_true, y_pred, max_val=1.)
        return ssim

    return masked_ssim

def psnr(dim, paddings=None, apply_idct=False, dct_min=None, dct_max=None):
    def masked_psnr(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        if apply_idct:
            min_tf = tf.constant(dct_min, dtype=tf.float16)
            max_tf = tf.constant(dct_max, dtype=tf.float16)
            dct_diff = (max_tf - min_tf)
            y_true = tf.cast((tf.cast(y_true, dtype=tf.float16) * dct_diff) + min_tf, dtype=tf.float32)
            y_pred = tf.cast((tf.cast(y_pred, dtype=tf.float16) * dct_diff) + min_tf, dtype=tf.float32)
            psnr = tf.image.psnr(image_batch_idct(y_true), image_batch_idct(y_pred), max_val=1.)
        else:
            psnr = tf.image.psnr(y_true, y_pred, max_val=1.)

        return psnr

    return masked_psnr

def mse(dim, paddings=None, weight=1, apply_idct=False, dct_min=None, dct_max=None):
    def masked_mse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        if apply_idct:
            min_tf = tf.constant(dct_min, dtype=tf.float16)
            max_tf = tf.constant(dct_max, dtype=tf.float16)
            dct_diff = (max_tf - min_tf)
            y_true = tf.cast((tf.cast(y_true, dtype=tf.float16) * dct_diff) + min_tf, dtype=tf.float32)
            y_pred = tf.cast((tf.cast(y_pred, dtype=tf.float16) * dct_diff) + min_tf, dtype=tf.float32)
            mse_val = tf.metrics.mean_squared_error(image_batch_idct(y_true), image_batch_idct(y_pred))
        else:
            mse_val = tf.metrics.mean_squared_error(y_pred, y_true)

        return weight * mse_val

    return masked_mse

def dct_and_pixelwise_mse(dim, paddings=None, mse_weight=1., dct_weight=1.):
    def masked_dct_and_mse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        dct_mse_val = tf.metrics.mean_squared_error(image_batch_dct(y_pred), image_batch_dct(y_true))
        mse_val = tf.metrics.mean_squared_error(y_true, y_pred)
        return (mse_weight * mse_val) + (dct_weight * dct_mse_val)

    return masked_dct_and_mse

def dct_and_pixelwise_mae(dim, paddings=None, mse_weight=1., dct_weight=1.):
    def masked_dct_and_mse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        dct_mse_val = tf.metrics.mean_absolute_error(image_batch_dct(y_pred), image_batch_dct(y_true))
        mse_val = tf.metrics.mean_squared_error(y_true, y_pred)
        return (mse_weight * mse_val) + (dct_weight * dct_mse_val)

    return masked_dct_and_mse

def blockwisedct_and_pixelwise_mse(dim, paddings=None, mse_weight=1., dct_weight=1.):
    def masked_dct_and_mse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        dct_mse_val = tf.metrics.mean_squared_error(dct_blockwise_batch(y_pred), dct_blockwise_batch(y_true))
        mse_val = tf.metrics.mean_squared_error(y_true, y_pred)
        return (mse_weight * mse_val) + (dct_weight * dct_mse_val)

    return masked_dct_and_mse

def blockwisedct_mse(dim, paddings=None, weight=1.):
    def masked_dct_and_mse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        dct_mse_val = tf.metrics.mean_squared_error(dct_blockwise_batch(y_pred), dct_blockwise_batch(y_true))
        return weight * dct_mse_val

    return masked_dct_and_mse

def dct_mse(dim, paddings=None, weight=1.):
    def masked_dctmse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        dct_mse_val = tf.metrics.mean_squared_error(image_batch_dct(y_pred), image_batch_dct(y_true))
        #mse_val = tf.metrics.mean_squared_error(y_true, y_pred)
        return (weight * dct_mse_val)

    return masked_dctmse

def dct_mae(dim, paddings=None, weight=1.):
    def masked_dctmse(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        dct_mse_val = tf.metrics.mean_absolute_error(image_batch_dct(y_pred), image_batch_dct(y_true))
        #mse_val = tf.metrics.mean_squared_error(y_true, y_pred)
        return (weight * dct_mse_val)

    return masked_dctmse

@tf.function
def image_batch_dct(image, norm='ortho'):
        """Does a type-II DCT (aka "The DCT") on axes 1 and 2 of a rank-4 tensor."""
        # b,h,w,c
        image = tf.transpose(image, perm=(0, 3, 1, 2))  # b,c,h,w
        dct_x = tf.signal.dct(image, norm=norm)
        dct_x = tf.transpose(dct_x, perm=[0, 1, 3, 2])  # b,c,w,h
        dct_y = tf.signal.dct(dct_x, norm=norm)
        dct_y = tf.transpose(dct_y, perm=[0, 3, 2, 1])  # b,h,w,c
        return dct_y

@tf.function
def image_batch_idct(dct, norm='ortho'):
        """Inverts image_batch_dct(), by performing a type-III DCT."""
        # b,h,w,c
        dct = tf.transpose(dct, perm=(0, 3, 1, 2))  # b,c,h,w
        idct = tf.signal.idct(dct, norm=norm)
        idct = tf.transpose(idct, perm=[0, 1, 3, 2])  # b,c,w,h
        idct = tf.signal.idct(idct, norm=norm)
        idct = tf.transpose(idct, perm=[0, 3, 2, 1])  # b,h,w,c
        return idct

@tf.function
def rotate(image: tf.Tensor, angle: tf.Tensor, fill_value: float = -1000.0):
    return tfa.image.rotate(image-fill_value, angle, interpolation='BILINEAR')+fill_value

@tf.function
def translate(image: tf.Tensor, fill_value: float = -1000.0):
    x = tf.random.uniform(minval=-50, maxval=50, dtype=tf.int32, shape=[])
    y = tf.random.uniform(minval=-50, maxval=50, dtype=tf.int32, shape=[])
    return tfa.image.translate(image-fill_value, [x, y], interpolation='BILINEAR')+fill_value

@tf.function
def flip(image: tf.Tensor):
    return tf.image.flip_left_right(image)

@tf.function
def flip_rotate(image: tf.Tensor, angle: tf.Tensor, fill_value: float = -1000.0):
    image = tf.image.flip_left_right(image)
    return tfa.image.rotate(image-fill_value, angle, interpolation='BILINEAR')+fill_value

@tf.function
def scale(image: tf.Tensor, ratio: tf.Tensor, dim=[384, 384, 1], fill_value: float = -1000.0):
   # print(ratio)
    x1 = y1 = 0.5 - (0.5 * ratio)
    x2 = y2 = 0.5 + (0.5 * ratio)
    boxes = [x1,y1,x2,y2]
    boxes = tf.cast(tf.reshape(boxes, shape=(1,4)), dtype=tf.float32)
    scaled = tf.image.crop_and_resize([image-fill_value], boxes=boxes, box_indices=tf.zeros(1, dtype=tf.int32),
                                        crop_size=(dim[0], dim[1]))+fill_value
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

def multiscale_ssim_l2(dim, mse_weight=1, ssim_weight=1, paddings=None):
    def masked_mssim_l2(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        ms_ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.)
        l2 = tf.reduce_mean(tf.keras.metrics.MSE(y_true, y_pred), axis=(-1, -2))
        loss = (ssim_weight * ms_ssim) + (mse_weight * l2)
        return loss

    return masked_mssim_l2

def ssim_l2(dim, mse_weight=1, ssim_weight=1, paddings=None):
    def masked_mssim_l2(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        ssim = 1 - tf.image.ssim(y_true, y_pred, max_val=1.)
        l2 = tf.reduce_mean(tf.keras.metrics.MSE(y_true, y_pred), axis=(-1, -2))
        loss = (ssim_weight * ssim) + (mse_weight * l2)
        return loss

    return masked_mssim_l2

def multiscale_ssim_l1(dim, mae_weight=1, ssim_weight=1, paddings=None):
    def masked_mssim_l1(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        ms_ssim = 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val=1.)
        l1 = tf.reduce_mean(tf.keras.metrics.mean_absolute_error(y_true, y_pred), axis=(-1, -2))
        loss = (ssim_weight * ms_ssim) + (mae_weight * l1)
        return loss

    return masked_mssim_l1



def laplacian_of_gaussian_MSE(dim, log_weight=1., mse_weight=1., log_max_weight=0.05, paddings=None):
    sigma = GAUSSIAN_SIGMA
    ksize = GAUSSIAN_LAPLACIAN_K_SIZE
    def log_mse_combined(y_true, y_pred):
            #log_weight   = next(iter_a)
            if paddings is not None:
                temp = tf.constant(value=1.0, shape=dim)
                mask = tf.pad(temp, paddings=paddings)
                y_true = mask * y_true
                y_pred = mask * y_pred
            log_p = laplacian(gaussian(y_pred, ksize=ksize, sigma=sigma), ksize=ksize)
            log_t = laplacian(gaussian(y_true, ksize=ksize, sigma=sigma), ksize=ksize)
            mse_log = tf.keras.losses.MSE(log_t, log_p)
            mse_log = tf.reduce_mean(tf.minimum((log_weight * mse_log), log_max_weight))
            l2 = tf.reduce_mean(tf.keras.metrics.MSE(y_true, y_pred), axis=(-1, -2))
            return (mse_weight * l2) + mse_log
    return log_mse_combined

def laplacian_of_gaussian_mae_MSE(dim, log_weight=0.1, mse_weight=1., log_max_weight=0.05, paddings=None):
    sigma = GAUSSIAN_SIGMA
    ksize = GAUSSIAN_LAPLACIAN_K_SIZE
    def log_mse_combined(y_true, y_pred):
            #log_weight   = next(iter_a)
            if paddings is not None:
                temp = tf.constant(value=1.0, shape=dim)
                mask = tf.pad(temp, paddings=paddings)
                y_true = mask * y_true
                y_pred = mask * y_pred
            log_p = laplacian(gaussian(y_pred, ksize=ksize, sigma=sigma), ksize=ksize)
            log_t = laplacian(gaussian(y_true, ksize=ksize, sigma=sigma), ksize=ksize)
            mse_log = tf.keras.losses.MAE(log_t, log_p)
            mse_log = tf.reduce_mean(tf.minimum((log_weight * mse_log), log_max_weight))
            l2 = tf.reduce_mean(tf.keras.metrics.MSE(y_true, y_pred), axis=(-1, -2))
            return (mse_weight * l2) + mse_log
    return log_mse_combined

def laplacian_of_gaussian_MAE(dim, log_weight=1., mae_weight=1., log_max_weight=0.05, paddings=None):
    sigma = GAUSSIAN_SIGMA
    ksize = GAUSSIAN_LAPLACIAN_K_SIZE
    def log_mse_combined(y_true, y_pred):
            #log_weight   = next(iter_a)
            if paddings is not None:
                temp = tf.constant(value=1.0, shape=dim)
                mask = tf.pad(temp, paddings=paddings)
                y_true = mask * y_true
                y_pred = mask * y_pred
            log_p = laplacian(gaussian(y_pred, ksize=ksize, sigma=sigma), ksize=ksize)
            log_t = laplacian(gaussian(y_true, ksize=ksize, sigma=sigma), ksize=ksize)
            mse_log = tf.keras.losses.MAE(log_t, log_p)
            mse_log = tf.reduce_mean(tf.minimum((log_weight * mse_log), log_max_weight))
            l1 = tf.reduce_mean(tf.keras.metrics.MAE(y_true, y_pred), axis=(-1, -2))
            return (mae_weight * l1) + mse_log
    return log_mse_combined


def laplacian_of_gaussian_mse_metric(dim, weight=1., paddings=None, log_max_weigt=0.05,
                                     sigma=GAUSSIAN_SIGMA, ksize=GAUSSIAN_LAPLACIAN_K_SIZE):
    def LoG(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        log_p = laplacian(gaussian(y_pred, ksize=ksize, sigma=sigma), ksize=ksize)
        log_t = laplacian(gaussian(y_true, ksize=ksize, sigma=sigma), ksize=ksize)
        mse_log = tf.keras.losses.MSE(log_t, log_p)
        mse_log = tf.reduce_mean(tf.minimum((weight * mse_log), log_max_weigt))
        return mse_log
    return LoG

def laplacian_of_gaussian_mae_metric(dim, weight=1., paddings=None, log_max_weigt=0.05,
                                     sigma=GAUSSIAN_SIGMA, ksize=GAUSSIAN_LAPLACIAN_K_SIZE):
    def LoG(y_true, y_pred):
        if paddings is not None:
            temp = tf.constant(value=1.0, shape=dim)
            mask = tf.pad(temp, paddings=paddings)
            y_true = mask * y_true
            y_pred = mask * y_pred
        log_p = laplacian(gaussian(y_pred, ksize=ksize, sigma=sigma), ksize=ksize)
        log_t = laplacian(gaussian(y_true, ksize=ksize, sigma=sigma), ksize=ksize)
        mse_log = tf.keras.losses.MAE(log_t, log_p)
        mse_log = tf.reduce_mean(tf.minimum((weight * mse_log), log_max_weigt))
        return mse_log
    return LoG

def dct_blockwise_batch(img, threshold_val=None): # threshold=0.012
    shape = tf.shape(img)
    b, x, y, c = shape[0], shape[1], shape[2], shape[3]
    img_res = tf.reshape(img, [b, x // 8, 8, y // 8, 8, c])
    img_dct1 = tf.signal.dct(tf.transpose(img_res, [0, 1, 2, 3, 5, 4]), norm='ortho')
    img_dct2 = tf.signal.dct(tf.transpose(img_dct1, [0, 1, 3, 5, 4, 2]), norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0, 1, 5, 2, 3, 4]), shape)
    if threshold_val is not None:
        mul = tf.math.scalar_mul(threshold_val, tf.math.reduce_max(out))
        out = tf.where(tf.math.greater(out, mul), out, 0)
    return out

def idct_blockwise_batch(img, thresholded=False, threshold_value=0.012):
    shape = tf.shape(img)
    b, x, y, c = shape[0], shape[1], shape[2], shape[3]
    print(x, y, c)
    img_res = tf.reshape(img, [b, x // 8, 8, y // 8, 8, c])
    img_dct1 = tf.signal.idct(tf.transpose(img_res, [0, 1, 2, 3, 5, 4]), norm='ortho')
    img_dct2 = tf.signal.idct(tf.transpose(img_dct1, [0, 1, 3, 5, 4, 2]), norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0, 1, 5, 2, 3, 4]), shape)
    if thresholded:
        mul = tf.math.scalar_mul(threshold_value, tf.math.reduce_max(out))
        out = tf.where(tf.math.greater(out, mul), out, 0)

    return out



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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

