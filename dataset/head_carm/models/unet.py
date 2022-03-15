from tensorflow.keras.layers import concatenate, Input, Conv2D, \
    Conv2DTranspose, Dropout, BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model


class UNet:
    kernel_init = 'he_normal'

    @staticmethod
    def down_layer(input_layer, filter_size, i, batch_norm=False, act='relu'):
        conv1 = Conv2D(
            filter_size[0],
            (3, 3),
            activation=act,
            padding='same',
            name=f'conv{i}_1',
            kernel_initializer=UNet.kernel_init)(input_layer)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(
            filter_size[1],
            (3, 3),
            strides=(2, 2),
            activation=act,
            padding='same',
            name=f'conv{i}_2',
            kernel_initializer=UNet.kernel_init)(conv1)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        return conv1, conv2

    @staticmethod
    def down_layer_with_reg(input_layer, filter_size, i, batch_norm=False, act='relu'):
        conv1 = Conv2D(
            filter_size[0],
            (3, 3),
            activation=act,
            padding='same',
            name=f'conv{i}_1',
            kernel_initializer=UNet.kernel_init,
            kernel_regularizer=L2(0.001))(input_layer)
        if batch_norm:
            conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(
            filter_size[1],
            (3, 3),
            strides=(2, 2),
            activation=act,
            padding='same',
            nname=f'conv{i}_2',
            kernel_initializer=UNet.kernel_init,
            kernel_regularizer=L2(0.001))(conv1)
        if batch_norm:
            conv2 = BatchNormalization()(conv2)

        return conv1, conv2

    @staticmethod
    def up_layer(input_layer, concat_layer, filter_size, i, batch_norm=False, dropout=False, act='relu'):
        up = Conv2DTranspose(
            filter_size//2,
            (2, 2),
            strides=(2, 2),
            activation=act,
            padding='same',
            name=f'up{i}',
            kernel_initializer=UNet.kernel_init)(input_layer)
        up = concatenate([up, concat_layer])
        conv = Conv2D(
            int(filter_size//2), (3, 3),
            activation=act, padding='same',
            name=f'conv{i}_1',
            kernel_initializer=UNet.kernel_init)(up)
        if batch_norm:
            conv = BatchNormalization()(conv)
        if dropout:
            conv = Dropout(0.5, seed=3, name='Dropout_' + str(i))(conv)

        return conv

    @staticmethod
    def up_layer_with_reg(input_layer, concat_layer, filter_size, i, batch_norm=False, dropout=False, act='relu'):
        up = Conv2DTranspose(
            filter_size//2, (2, 2),
            strides=(2, 2),
            activation=act,
            padding='same',
            name=f'up{i}',
            kernel_initializer=UNet.kernel_init,
            kernel_regularizer=L2(0.001))(input_layer)

        up = concatenate([up, concat_layer])
        conv = Conv2D(
            int(filter_size//2), (3, 3),
            activation=act, padding='same',
            name=f'conv{i}_1',
            kernel_initializer=UNet.kernel_init,
            kernel_regularizer=L2(0.001))(up)
        if batch_norm:
            conv = BatchNormalization()(conv)
        if dropout:
            conv = Dropout(0.5, seed=3, name=f'Dropout_{i}')(conv)

        return conv

    @staticmethod
    def build_model(img_shape=(384, 384, 1), act='relu', d=16, batch_norm=False, dropout=False):
        input_img = Input(img_shape, name='img_inp')
        sfs = d  # start filter size

        conv0, conv1 = UNet.down_layer(input_img, [sfs, sfs*2], 1, batch_norm=batch_norm, act=act)  # 16, 32
        _, conv2 = UNet.down_layer(conv1, [sfs*4, sfs*8], 2, batch_norm=batch_norm, act=act)  # 64, 128
        _, conv3 = UNet.down_layer(conv2, [sfs*16, sfs*32], 3, batch_norm=batch_norm, act=act)  # 128, 256
        conv4 = Conv2D(
            sfs*32, (3, 3),
            activation=act,
            strides=(2, 2),
            padding='same',
            name='conv5',
            kernel_initializer=UNet.kernel_init)(conv3)  # 256
        # if bn:
        #     conv4 = BatchNormalization()(conv4)

        conv6 = UNet.up_layer(conv4, conv3, sfs * 16, 6, batch_norm=batch_norm, dropout=dropout, act=act)
        conv7 = UNet.up_layer(conv6, conv2, sfs * 8, 7, batch_norm=batch_norm, dropout=dropout, act=act)
        conv8 = UNet.up_layer(conv7, conv1, sfs * 4, 8, batch_norm=batch_norm, dropout=dropout, act=act)
        conv9 = UNet.up_layer(conv8, conv0, sfs * 2, 9, batch_norm=batch_norm, dropout=dropout, act=act)
        # conv10 = self.upLayer(conv9, conv0, sfs , 10, bn=bn, do=do, act=act)

        conv_out = Conv2D(1, (1, 1), activation=act, name='conv_final')(conv9)
        p_model = Model(input_img, conv_out)

        return p_model

    @staticmethod
    def build_l2_model(img_shape=(384, 384, 1), act='relu', d=16, batch_norm=False, dropout=False):
        input_img = Input(img_shape, name='img_inp')
        sfs = d  # start filter size

        conv0, conv1 = UNet.down_layer_with_reg(input_img, [sfs, sfs*2], 1, batch_norm=batch_norm, act=act)  # 16, 32
        _, conv2 = UNet.down_layer_with_reg(conv1, [sfs*4, sfs*8], 2, batch_norm=batch_norm, act=act)  # 64, 128
        _, conv3 = UNet.down_layer_with_reg(conv2, [sfs*16, sfs*32], 3, batch_norm=batch_norm, act=act)  # 128, 256
        conv4 = Conv2D(
            sfs*32,
            (3, 3),
            activation=act,
            strides=(2, 2),
            padding='same',
            name='conv5',
            kernel_initializer=UNet.kernel_init,
            kernel_regularizer=L2(0.001))(conv3)  # 256
        conv6 = UNet.up_layer_with_reg(conv4, conv3, sfs * 16, 6, batch_norm=batch_norm, dropout=dropout, act=act)
        conv7 = UNet.up_layer_with_reg(conv6, conv2, sfs * 8, 7, batch_norm=batch_norm, dropout=dropout, act=act)
        conv8 = UNet.up_layer_with_reg(conv7, conv1, sfs * 4, 8, batch_norm=batch_norm, dropout=dropout, act=act)
        conv9 = UNet.up_layer_with_reg(conv8, conv0, sfs * 2, 9, batch_norm=batch_norm, dropout=dropout, act=act)
        # conv10 = self.upLayer(conv9, conv0, sfs , 10, bn=bn, do=do, act=act)

        conv_out = Conv2D(
            1,
            (1, 1),
            activation=act,
            name='conv_final',
            kernel_initializer=UNet.kernel_init,
            kernel_regularizer=L2(0.001))(conv9)
        p_model = Model(input_img, conv_out)

        return p_model
