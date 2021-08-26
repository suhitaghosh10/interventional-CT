
from tensorflow.keras.layers import concatenate, Input, Conv2D, Conv2DTranspose, \
    Dropout, BatchNormalization, Lambda

from tensorflow.keras.models import Model


class unet:
    def __init__(self):

        super(unet, self).__init__()
        self.kernel_init = 'he_normal'

    def downLayer(self, inputLayer, filterSize, name, bn=False, act='relu'):

        conv1 = Conv2D(filterSize[0], (3, 3), activation=act, padding='same', name='conv' + str(name) + '_1', kernel_initializer=self.kernel_init)(inputLayer)
        if bn:
            conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filterSize[1], (3, 3), strides=(2,2), activation=act, padding='same', name='conv' + str(name) + '_2', kernel_initializer=self.kernel_init)(conv1)
        if bn:
            conv2 = BatchNormalization()(conv2)

        return conv2

    def upLayer(self, inputLayer, filterSize, name, bn=False, act='relu', do=False):

        conv1 = Conv2D(filterSize[0], (3, 3), activation=act, padding='same', name='conv' + str(name) + '_1', kernel_initializer=self.kernel_init)(inputLayer)
        if bn:
            conv1 = BatchNormalization()(conv1)
        if do:
            conv1 = Dropout(0.5, seed=3, name='Dropout_' + str(name))(conv1)
        conv2 = Conv2DTranspose(filterSize[1], (3, 3), strides=(2,2), activation=act, padding='same', name='deconv' + str(name) + '_2', kernel_initializer=self.kernel_init)(conv1)
        if bn:
            conv2 = BatchNormalization()(conv2)
        if do:
            conv2 = Dropout(0.5, seed=3, name='Dropout_' + str(name))(conv2)

        return conv2

    def build_model(self, img_shape=(2, 384, 384, 1), act='relu', d=4, bn=False, do=False):

        input = Input(img_shape, name='input')
        prior_input = Lambda(lambda x: x[:, 0, :, :, :])(input)
        cbct_input = Lambda(lambda x: x[:, 1, :, :, :])(input)
        sfs = d  # start filter size

        # PRIOR
        conv_prior_1 = self.downLayer(prior_input, filterSize=[sfs, sfs*2], name='p1', bn=bn,act=act)
        conv_prior_2 = self.downLayer(conv_prior_1, filterSize=[sfs * 2, sfs * 4], name='p2', bn=bn, act=act)
        conv_prior_3 = self.downLayer(conv_prior_2, filterSize=[sfs * 4, sfs * 8], name='p3', bn=bn, act=act)
        conv_prior_4 = self.downLayer(conv_prior_3, filterSize=[sfs * 8, sfs * 16], name='p4', bn=bn, act=act)
        conv_prior_5 = self.downLayer(conv_prior_4, filterSize=[sfs * 16, sfs * 32], name='p5', bn=bn, act=act)

        # CBCT
        conv_cbct_1 = self.downLayer(cbct_input, filterSize=[sfs, sfs * 2], name='c1', bn=bn, act=act)
        conv_cbct_2 = self.downLayer(conv_cbct_1, filterSize=[sfs * 2, sfs * 4], name='c2', bn=bn, act=act)
        conv_cbct_3 = self.downLayer(conv_cbct_2, filterSize=[sfs * 4, sfs * 8], name='c3', bn=bn, act=act)
        conv_cbct_4 = self.downLayer(conv_cbct_3, filterSize=[sfs * 8, sfs * 16], name='c4', bn=bn, act=act)
        conv_cbct_5 = self.downLayer(conv_cbct_4, filterSize=[sfs * 16, sfs * 32], name='c5', bn=bn, act=act)

        concat1 = concatenate([conv_prior_5, conv_cbct_5])
        concat1_shape = concat1.shape
        deconv1 = self.upLayer(concat1, [concat1_shape[-1], sfs * 32], 'up1', bn=bn, do=do, act=act)

        concat2 = concatenate([deconv1, conv_prior_4, conv_cbct_4])
        deconv2 = self.upLayer(concat2, [sfs * 16, sfs * 16], 'up2', bn=bn, do=do, act=act)

        concat3 = concatenate([deconv2, conv_prior_3, conv_cbct_3])
        deconv3 = self.upLayer(concat3, [sfs * 8, sfs * 8], 'up3', bn=bn, do=do, act=act)

        concat4 = concatenate([deconv3, conv_prior_2, conv_cbct_2])
        deconv4 = self.upLayer(concat4, [sfs * 4, sfs * 4], 'up4', bn=bn, do=do, act=act)

        concat5 = concatenate([deconv4, conv_prior_1, conv_cbct_1])
        deconv5 = self.upLayer(concat5, [sfs * 2, sfs * 2], 'up5', bn=bn, do=do, act=act)

        conv = Conv2D(sfs, (3, 3), activation=act, padding='same', name='up61',
                       kernel_initializer=self.kernel_init)(deconv5)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3)(conv)
        conv = Conv2D(sfs, (3, 3), activation=act, padding='same', name='up62',
                       kernel_initializer=self.kernel_init)(conv)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3)(conv)
        conv = Conv2D(1, (3, 3), activation=act, padding='same', name='last_conv',
                      kernel_initializer=self.kernel_init)(conv)

        p_model = Model(inputs=input, outputs=conv)

        return p_model

