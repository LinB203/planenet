from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, \
    DepthwiseConv2D, Reshape, GlobalAveragePooling2D, Add, Activation, \
    BatchNormalization, Flatten, Dropout, Input, Multiply, Lambda, Concatenate, LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2

leaky_alpha = 0.1
c = 2
base_c1 = True
def Conv_block(x, filters):

    x = DepthwiseConv2D((3, 3), padding='same', kernel_regularizer=l2(3e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(leaky_alpha)(x)
    x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(3e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(leaky_alpha)(x)
    x = DepthwiseConv2D((3, 3), padding='same', kernel_regularizer=l2(3e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(leaky_alpha)(x)
    return x

def channel_split_conv(x, double_filters=True):
    if base_c1:
        if not double_filters:
            c_interval_1_conv = Lambda(lambda z: z[:, :, :, ::c])(x)
            c_interval_1_no_conv = Lambda(lambda z: z[:, :, :, 1::c])(x)
            filters = c_interval_1_conv.shape.as_list()[-1]
            c_interval_1_after_conv = Conv_block(c_interval_1_conv, filters)
            c_1 = Concatenate()([c_interval_1_after_conv, c_interval_1_no_conv])

            c_interval_2_conv = Lambda(lambda z: z[:, :, :, ::c+1])(c_1)
            c_interval_2_no_conv_ = Lambda(lambda z: z[:, :, :, 1::c+1])(c_1)
            c_interval_2_no_conv__ = Lambda(lambda z: z[:, :, :, 2::c+1])(c_1)
            c_interval_2_no_conv = Concatenate()([c_interval_2_no_conv_, c_interval_2_no_conv__])
            filters = c_interval_2_conv.shape.as_list()[-1]
            c_interval_2_after_conv = Conv_block(c_interval_2_conv, filters)
            c_2 = Concatenate()([c_interval_2_after_conv, c_interval_2_no_conv])
        else:
            c_interval_1_conv = Lambda(lambda z: z[:, :, :, ::c])(x)
            c_interval_1_no_conv = Lambda(lambda z: z[:, :, :, 1::c])(x)
            filters = c_interval_1_conv.shape.as_list()[-1]
            c_interval_1_after_conv = Conv_block(c_interval_1_conv, filters)
            c_1 = Concatenate()([c_interval_1_after_conv, c_interval_1_no_conv])

            c_interval_2_conv = Lambda(lambda z: z[:, :, :, ::c + 1])(c_1)
            c_interval_2_no_conv_ = Lambda(lambda z: z[:, :, :, 1::c + 1])(c_1)
            c_interval_2_no_conv__ = Lambda(lambda z: z[:, :, :, 2::c + 1])(c_1)
            c_interval_2_no_conv = Concatenate()([c_interval_2_no_conv_, c_interval_2_no_conv__])
            filters = c_interval_2_conv.shape.as_list()[-1]
            c_interval_2_after_conv = Conv_block(c_interval_2_conv, filters)
            c_2 = Concatenate()([c_1, c_interval_2_after_conv, c_interval_2_no_conv])
    else:
        if not double_filters:
            c_interval_1_conv = Lambda(lambda z: z[:, :, :, ::c])(x)
            c_interval_1_no_conv = Lambda(lambda z: z[:, :, :, 1::c])(x)
            filters = c_interval_1_conv.shape.as_list()[-1]
            c_interval_1_after_conv = Conv_block(c_interval_1_conv, filters)
            c_1 = Concatenate()([c_interval_1_after_conv, c_interval_1_no_conv])

            c_interval_2_conv = Lambda(lambda z: z[:, :, :, ::c + 1])(x)
            c_interval_2_no_conv_ = Lambda(lambda z: z[:, :, :, 1::c + 1])(x)
            c_interval_2_no_conv__ = Lambda(lambda z: z[:, :, :, 2::c + 1])(x)
            c_interval_2_no_conv = Concatenate()([c_interval_2_no_conv_, c_interval_2_no_conv__])
            filters = c_interval_2_conv.shape.as_list()[-1]
            c_interval_2_after_conv = Conv_block(c_interval_2_conv, filters)
            c_2 = Add()([c_1, Concatenate()([c_interval_2_after_conv, c_interval_2_no_conv])])
        else:
            c_interval_1_conv = Lambda(lambda z: z[:, :, :, ::c])(x)
            c_interval_1_no_conv = Lambda(lambda z: z[:, :, :, 1::c])(x)
            filters = c_interval_1_conv.shape.as_list()[-1]
            c_interval_1_after_conv = Conv_block(c_interval_1_conv, filters)
            c_1 = Concatenate()([c_interval_1_after_conv, c_interval_1_no_conv])

            c_interval_2_conv = Lambda(lambda z: z[:, :, :, ::c + 1])(x)
            c_interval_2_no_conv_ = Lambda(lambda z: z[:, :, :, 1::c + 1])(x)
            c_interval_2_no_conv__ = Lambda(lambda z: z[:, :, :, 2::c + 1])(x)
            c_interval_2_no_conv = Concatenate()([c_interval_2_no_conv_, c_interval_2_no_conv__])
            filters = c_interval_2_conv.shape.as_list()[-1]
            c_interval_2_after_conv = Conv_block(c_interval_2_conv, filters)
            c_2 = Concatenate()([c_1, c_interval_2_after_conv, c_interval_2_no_conv])
    return c_2



def split_VGG16(input_shape, classes):
    img_input = Input(shape=input_shape)
    x = img_input
    # 第一个卷积部分
    # 112，112，64
    x = Conv2D(32, (1, 1), padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = channel_split_conv(x, double_filters=False)
    # x = channel_split_conv(x, double_filters=False)
    # x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # 第二个卷积部分
    # 56,56,128
    x = channel_split_conv(x, double_filters=True)
    x = channel_split_conv(x, double_filters=False)
    # x = channel_split_conv(x, double_filters=False)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # 第三个卷积部分
    # 28,28,256
    x = channel_split_conv(x, double_filters=True)
    x = channel_split_conv(x, double_filters=False)
    # x = channel_split_conv(x, double_filters=False)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # 第四个卷积部分
    # 14,14,512
    x = channel_split_conv(x, double_filters=True)
    x = channel_split_conv(x, double_filters=False)
    # x = channel_split_conv(x, double_filters=False)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # 第五个卷积部分
    # 7,7,512
    x = channel_split_conv(x, double_filters=False)
    x = channel_split_conv(x, double_filters=False)
    # x = channel_split_conv(x, double_filters=False)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # 提取特征

    # 分类部分
    # 7x7x512
    # 25088
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(img_input, x, name='vgg16')

    return model
