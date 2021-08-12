from tensorflow.keras.layers import concatenate, Input, Conv2D, Conv2DTranspose, \
    Dropout, BatchNormalization
from tensorflow.keras.regularizers import L2
from keras.models import Model

class unet:
    def __init__(self):

        super(unet, self).__init__()
        self.kernel_init = 'he_normal'

    def downLayer(self, inputLayer, filterSize, i, bn=False, act='relu'):

        conv1 = Conv2D(filterSize[0], (3, 3), activation=act, padding='same', name='conv' + str(i) + '_1', kernel_initializer=self.kernel_init)(inputLayer)
        if bn:
            conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filterSize[1], (3, 3), strides=(2,2), activation=act, padding='same', name='conv' + str(i) + '_2', kernel_initializer=self.kernel_init)(conv1)
        if bn:
            conv2 = BatchNormalization()(conv2)

        return conv1, conv2

    def downLayer_with_reg(self, inputLayer, filterSize, i, bn=False, act='relu'):

        conv1 = Conv2D(filterSize[0],
                       (3, 3),
                       activation=act,
                       padding='same',
                       name='conv' + str(i) + '_1',
                       kernel_initializer=self.kernel_init,
                       kernel_regularizer=L2(0.001))(inputLayer)
        if bn:
            conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filterSize[1],
                       (3, 3),
                       strides=(2,2),
                       activation=act,
                       padding='same',
                       name='conv' + str(i) + '_2',
                       kernel_initializer=self.kernel_init,
                       kernel_regularizer=L2(0.001))(conv1)
        if bn:
            conv2 = BatchNormalization()(conv2)

        return conv1, conv2

    def upLayer(self, inputLayer, concatLayer, filterSize, i, bn=False, do=False, act='relu'):
        up = Conv2DTranspose(filterSize//2, (2, 2), strides=(2, 2), activation=act, padding='same',
                             name='up' + str(i), kernel_initializer=self.kernel_init)(inputLayer)
        # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv2D(int(filterSize//2), (3, 3),
                      activation=act, padding='same',
                      name='conv' + str(i) + '_1',
                      kernel_initializer=self.kernel_init)(
            up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3, name='Dropout_' + str(i))(conv)

        return conv

    def upLayer_with_reg(self, inputLayer, concatLayer, filterSize, i, bn=False, do=False, act='relu'):

        up = Conv2DTranspose(filterSize//2, (2, 2),
                             strides=(2, 2),
                             activation=act,
                             padding='same',
                             name='up' + str(i),
                             kernel_initializer=self.kernel_init,
                             kernel_regularizer=L2(0.001)
                             )(inputLayer)

        up = concatenate([up, concatLayer])
        conv = Conv2D(int(filterSize//2), (3, 3),
                      activation=act, padding='same',
                      name='conv' + str(i) + '_1',
                      kernel_initializer=self.kernel_init,
                      kernel_regularizer=L2(0.001))(
            up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3, name='Dropout_' + str(i))(conv)

        return conv

    def build_model(self, img_shape=(384, 384, 1), act='relu', d=16, bn=False, do=False):

        input_img = Input(img_shape, name='img_inp')
        sfs = d  # start filter size

        conv0, conv1 = self.downLayer(input_img, [sfs, sfs*2], 1, bn=bn,act=act) #16, 32
        _, conv2 = self.downLayer(conv1, [sfs*4, sfs*8], 2, bn=bn,act=act) #64, 128
        _, conv3 = self.downLayer(conv2, [sfs *16, sfs*32], 3, bn=bn, act=act) #128, 256
        conv4 = Conv2D(sfs*32, (3, 3), activation=act, strides=(2,2), padding='same', name='conv5',
                       kernel_initializer=self.kernel_init)(conv3)#256
        # if bn:
        #     conv4 = BatchNormalization()(conv4)

        conv6 = self.upLayer(conv4, conv3, sfs * 16, 6, bn=bn, do=do, act=act)
        conv7 = self.upLayer(conv6, conv2, sfs * 8, 7, bn=bn, do=do, act=act)
        conv8 = self.upLayer(conv7, conv1, sfs * 4, 8, bn=bn, do=do, act=act)
        conv9 = self.upLayer(conv8, conv0, sfs * 2, 9, bn=bn, do=do, act=act)
        #conv10 = self.upLayer(conv9, conv0, sfs , 10, bn=bn, do=do, act=act)

        conv_out = Conv2D(1, (1, 1), activation=act, name='conv_final')(conv9)
        p_model = Model(input_img, conv_out)

        return p_model

    def build_l2_model(self, img_shape=(384, 384, 1), act='relu', d=16, bn=False, do=False):

        input_img = Input(img_shape, name='img_inp')
        sfs = d  # start filter size

        conv0, conv1 = self.downLayer_with_reg(input_img, [sfs, sfs*2], 1, bn=bn,act=act) #16, 32
        _, conv2 = self.downLayer_with_reg(conv1, [sfs*4, sfs*8], 2, bn=bn,act=act) #64, 128
        _, conv3 = self.downLayer_with_reg(conv2, [sfs *16, sfs*32], 3, bn=bn, act=act) #128, 256
        conv4 = Conv2D(sfs*32, (3, 3), activation=act, strides=(2,2), padding='same', name='conv5',
                       kernel_initializer=self.kernel_init, kernel_regularizer=L2(0.001))(conv3)#256
        conv6 = self.upLayer_with_reg(conv4, conv3, sfs * 16, 6, bn=bn, do=do, act=act)
        conv7 = self.upLayer_with_reg(conv6, conv2, sfs * 8, 7, bn=bn, do=do, act=act)
        conv8 = self.upLayer_with_reg(conv7, conv1, sfs * 4, 8, bn=bn, do=do, act=act)
        conv9 = self.upLayer_with_reg(conv8, conv0, sfs * 2, 9, bn=bn, do=do, act=act)
        #conv10 = self.upLayer(conv9, conv0, sfs , 10, bn=bn, do=do, act=act)

        conv_out = Conv2D(1, (1, 1),
                          activation=act,
                          name='conv_final',
                          kernel_initializer=self.kernel_init,
                          kernel_regularizer=L2(0.001))(conv9)
        p_model = Model(input_img, conv_out)

        return p_model
