from utility.common_imports import *
from matplotlib import pyplot as plt
from utility.constants import *
from utility.utils import augment_prior
from dataset.head_carm.constants import CARM_DCT_MIN, CARM_DCT_MAX
import io


class PlotReconstructionCallback(tfk.callbacks.Callback):
    """Plot reconstructed images to tensorboard."""

    def __init__(self, logdir: str, test_ds: tf.data.Dataset,
                 batch_size=4,
                 model_type=None,
                 save_by='val_loss',
                 save_by_decrease=True,
                 batched=False,
                 is_shuffle=False,
                 shuffle_buffer_size=128,
                 is_augment=True,
                 log_on_epoch_end=True,
                 step_num=100,
                 chkpoint_path=None,
                 init_epoch=1):

        super(PlotReconstructionCallback, self).__init__()
        self.model_type = model_type
        self.logdir = logdir
        self.save_by_key = save_by
        self.save_by_decrease = save_by_decrease
        self.batch_size = batch_size
        self.epoch_num = init_epoch + 1
        self.chkpoint_path = chkpoint_path
        self.log_on_epoch_end = log_on_epoch_end
        self.save_by_val = tf.constant(np.inf, dtype=tf.float64) if self.save_by_decrease else tf.constant(-np.inf,
                                                                                                           dtype=tf.float64)
        if not log_on_epoch_end:
            self.steps = step_num

        _logdir_recons = os.path.join(logdir, 'reconstructions')
        self.file_writer = tf.summary.create_file_writer(logdir=_logdir_recons)
        # dataset stuff
        if is_shuffle:
            test_ds = test_ds.shuffle(shuffle_buffer_size, seed=123)
        if is_augment:
            test_ds = test_ds.map(augment_prior)
        if batched:
            self.test_ds = test_ds.unbatch()
        self.test_ds = test_ds.batch(batch_size)
        self.test_it_ds = iter(self.test_ds)

    def on_epoch_end(self, epoch, logs=None):
        if tf.equal(epoch, 0):
            self.model.summary()
        if self.log_on_epoch_end:
            self._log_images(epoch)

        self._save_wts(logs)
        self.epoch_num += 1

    def on_batch_end(self, batch, logs=None):
        if (not self.log_on_epoch_end) and (batch % self.steps == 0):
            self._log_images((batch + 1) * (self.epoch_num))
            print('\nLogging for step no {} and epoch {}.'.format(batch + 1, self.epoch_num))

    def _get_next_images(self):
        try:
            next_images_x, next_images_y = next(self.test_it_ds)
        except StopIteration:
            self.test_it_ds = iter(self.test_ds)
            next_images_x, next_images_y = next(self.test_it_ds)
        return next_images_x, next_images_y

    def _plot_img_reconstruction(self, images, gt, reconstruction):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(30, 30)
        image = images[0]
        p_image = images[1]
        if self.model_type is MODEL_TYPE_DCT:
            dct_diff = (CARM_DCT_MAX - CARM_DCT_MIN)
            image = (image * dct_diff) + CARM_DCT_MIN
            gt = (gt * dct_diff) + CARM_DCT_MIN
            reconstruction = (reconstruction * dct_diff) + CARM_DCT_MIN

            image = idct_blockwise(tf.cast(image, tf.float32))
            gt = idct_blockwise(tf.cast(gt, tf.float32))
            reconstruction = idct_blockwise(tf.cast(reconstruction, tf.float32))
            # reconstruction = (reconstruction - tf.reduce_min(reconstruction)) \
            #                  / (tf.reduce_max(reconstruction) - tf.reduce_min(reconstruction))

        if image.shape[-1] == 1:
            image = tf.cast(tf.squeeze(image, axis=-1), dtype=tf.float32)
            reconstruction = tf.cast(tf.squeeze(reconstruction, axis=-1), dtype=tf.float32)
            gt = tf.cast(tf.squeeze(gt, axis=-1), dtype=tf.float32)

        # if len(img_shape)> 3: # 3D data
        #     ax[0, 0].imshow(image[:,:,img_shape[2]//2], vmin=0., vmax=1., cmap=plt.cm.gray)
        # elif img_shape[-1] == 1:
        #     ax[0,0].imshow(image, cmap=plt.cm.gray)
        # else:
        #     ax[0,0].imshow(image, vmin=0., vmax=1.)
        diff = tf.abs(gt - reconstruction)

        # txt = str(np.min(image))+str(np.max(image))
        # ax[0,0].set_title('Image'+txt)
        # mset = mse(image, gt).numpy()
        # ssimt = ssim(image, gt).numpy()
        # psnrt = psnr(image, gt).numpy()
        # text = str(mset)+','+str(ssimt)+','+str(psnrt)
        # ax[0,0].set_title(text)
        ax[0, 0].imshow(image, cmap=plt.cm.gray, vmin=0., vmax=1.)
        ax[0, 0].axis('off')
        ax[0, 1].imshow(reconstruction, cmap=plt.cm.gray, vmin=0., vmax=1.)
        # ax[0, 1].xlabel (text, fontsize=24)
        ax[0, 1].axis('off')
        ax[1, 0].imshow(gt, cmap=plt.cm.gray, vmin=0., vmax=1.)
        ax[1, 0].axis('off')
        ax[1, 1].imshow(p_image, vmin=0., vmax=1., cmap=plt.cm.gray)
        ax[1, 1].axis('off')

        return fig

    def _log_images(self, step_num):
        if self.batch_size < 4:
            batch_num = self.batch_size // 4
        else:
            batch_num = 1

        for b in range(batch_num):
            if b == 0:
                images, gts = self._get_next_images()
                reconstructions = self.model(images)
            else:
                temp_img, temp_gt = self._get_next_images()
                temp_recons = self.model(temp_img)
                images = tf.concat([images, temp_img], axis=0)
                gts = tf.concat([gts, temp_gt], axis=0)
                reconstructions = tf.concat([reconstructions, temp_recons], axis=0)

        imgs = []
        try:
            for i in range(self.batch_size):
                fig = self._plot_img_reconstruction(images[i], gts[i], reconstructions[i])
                imgs.append(_plot_to_image(fig))
            imgs = tf.concat(imgs, axis=0)
            with self.file_writer.as_default():
                tf.summary.image(
                    name='Reconstructions',
                    data=imgs,
                    step=step_num,
                    max_outputs=self.batch_size
                )
                # tf.summary.scalar('lr', self.model.optimizer.lr, step=step_num)
        except:
            traceback.print_exc()

    def _save_wts(self, logs):

        save_by_condn = self.save_by_val > logs[self.save_by_key] if self.save_by_decrease else self.save_by_val < logs[
            self.save_by_key]
        indc = 'decreased' if self.save_by_decrease else 'increased'
        if self.save_by_key in logs and save_by_condn:
            print('\n', self.save_by_key, indc, 'from', self.save_by_val, 'to', logs[self.save_by_key])
            self.save_by_val = logs[self.save_by_key]
            self.model.save_weights(self.chkpoint_path + '.hdf5')
            print("Saved checkpoint for epoch {}: {}".format(str(self.epoch_num), self.chkpoint_path))


def _plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
