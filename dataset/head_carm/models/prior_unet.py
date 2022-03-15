
from tensorflow.keras.layers import concatenate, Input, Conv2D, \
    Conv2DTranspose, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model


class UNet:
    kernel_init = 'he_normal'

    @staticmethod
    def down_layer(input_layer, filter_size, name, batch_norm=False, act='relu'):
        conv1 = Conv2D(
            filter_size[0],
            (3, 3),
            activation=act,
            padding='same',
            name=f'conv{name}_1',
            kernel_initializer=UNet.kernel_init)(input_layer)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(
            filter_size[1],
            (3, 3),
            strides=(2, 2),
            activation=act,
            padding='same',
            name=f'conv{name}_2',
            kernel_initializer=UNet.kernel_init)(conv1)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        return conv2

    @staticmethod
    def up_layer(input_layer, filter_size, name, batch_norm=False, act='relu', dropout=False):
        conv1 = Conv2D(
            filter_size[0],
            (3, 3),
            activation=act,
            padding='same',
            name=f'conv{name}_1',
            kernel_initializer=UNet.kernel_init)(input_layer)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        if dropout:
            conv1 = Dropout(0.5, seed=3, name=f'Dropout_{name}')(conv1)
        conv2 = Conv2DTranspose(
            filter_size[1],
            (3, 3),
            strides=(2, 2),
            activation=act,
            padding='same',
            name=f'deconv{name}_2',
            kernel_initializer=UNet.kernel_init)(conv1)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)
        if dropout:
            conv2 = Dropout(0.5, seed=3, name=f'Dropout_{name}')(conv2)

        return conv2

    @staticmethod
    def build_model(img_shape=(2, 384, 384, 1), act='relu', d=4, batch_norm=False, dropout=False):
        input_layer = Input(img_shape, name='input')
        prior_input = Lambda(lambda x: x[:, 0, :, :, :])(input_layer)
        cbct_input = Lambda(lambda x: x[:, 1, :, :, :])(input_layer)
        sfs = d  # start filter size

        # PRIOR
        conv_prior_1 = UNet.down_layer(prior_input, filter_size=[sfs, sfs*2], name='p1', batch_norm=batch_norm, act=act)  # 8, 16
        conv_prior_2 = UNet.down_layer(conv_prior_1, filter_size=[sfs * 2, sfs * 4], name='p2', batch_norm=batch_norm, act=act)  # 16, 32
        conv_prior_3 = UNet.down_layer(conv_prior_2, filter_size=[sfs * 4, sfs * 8], name='p3', batch_norm=batch_norm, act=act)  # 32, 64
        conv_prior_4 = UNet.down_layer(conv_prior_3, filter_size=[sfs * 8, sfs * 16], name='p4', batch_norm=batch_norm, act=act)  # 64, 128
        conv_prior_5 = UNet.down_layer(conv_prior_4, filter_size=[sfs * 16, sfs * 32], name='p5', batch_norm=batch_norm, act=act)  # 128, 256

        # CBCT
        conv_cbct_1 = UNet.down_layer(cbct_input, filter_size=[sfs, sfs * 2], name='c1', batch_norm=batch_norm, act=act)  # 8, 16
        conv_cbct_2 = UNet.down_layer(conv_cbct_1, filter_size=[sfs * 2, sfs * 4], name='c2', batch_norm=batch_norm, act=act)  # 16, 32
        conv_cbct_3 = UNet.down_layer(conv_cbct_2, filter_size=[sfs * 4, sfs * 8], name='c3', batch_norm=batch_norm, act=act)  # 32, 64
        conv_cbct_4 = UNet.down_layer(conv_cbct_3, filter_size=[sfs * 8, sfs * 16], name='c4', batch_norm=batch_norm, act=act)  # 64, 128
        conv_cbct_5 = UNet.down_layer(conv_cbct_4, filter_size=[sfs * 16, sfs * 32], name='c5', batch_norm=batch_norm, act=act)  # 128, 256

        concat1 = concatenate([conv_prior_5, conv_cbct_5])  # 256+256=512
        concat1_shape = concat1.shape
        deconv1 = UNet.up_layer(concat1, [concat1_shape[-1], sfs * 32], 'up1', batch_norm=batch_norm, dropout=dropout, act=act)  # 256

        concat2 = concatenate([deconv1, conv_prior_4, conv_cbct_4])
        deconv2 = UNet.up_layer(concat2, [sfs * 16, sfs * 16], 'up2', batch_norm=batch_norm, dropout=dropout, act=act)

        concat3 = concatenate([deconv2, conv_prior_3, conv_cbct_3])
        deconv3 = UNet.up_layer(concat3, [sfs * 8, sfs * 8], 'up3', batch_norm=batch_norm, dropout=dropout, act=act)

        concat4 = concatenate([deconv3, conv_prior_2, conv_cbct_2])
        deconv4 = UNet.up_layer(concat4, [sfs * 4, sfs * 4], 'up4', batch_norm=batch_norm, dropout=dropout, act=act)

        concat5 = concatenate([deconv4, conv_prior_1, conv_cbct_1])
        deconv5 = UNet.up_layer(concat5, [sfs * 2, sfs * 2], 'up5', batch_norm=batch_norm, dropout=dropout, act=act)

        conv = Conv2D(
            sfs,
            (3, 3),
            activation=act,
            padding='same',
            name='up61',
            kernel_initializer=UNet.kernel_init)(deconv5)
        if batch_norm:
            conv = BatchNormalization()(conv)
        if dropout:
            conv = Dropout(0.5, seed=3)(conv)
        conv = Conv2D(
            sfs,
            (3, 3),
            activation=act,
            padding='same',
            name='up62',
            kernel_initializer=UNet.kernel_init)(conv)
        if batch_norm:
            conv = BatchNormalization()(conv)
        if dropout:
            conv = Dropout(0.5, seed=3)(conv)
        conv = Conv2D(
            1,
            (3, 3),
            activation=act,
            padding='same',
            name='last_conv',
            kernel_initializer=UNet.kernel_init)(conv)

        p_model = Model(inputs=input_layer, outputs=conv)

        return p_model
