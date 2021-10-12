from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, \
    DepthwiseConv2D, Reshape, GlobalAveragePooling2D, Add, Activation, \
    BatchNormalization, Flatten, Dropout, Input, Multiply, Lambda, Concatenate, AvgPool2D, regularizers, LeakyReLU
from keras.models import Model
from keras import backend as K

base_filter = 32
radio = 1.0
l2 = 1e-4
l_alpha = 0.5

def se_block(x, radio=1.0):
    # input_channel = K.int_shape(x)[-1]
    # y = AvgPool2D(pool_size=(int(x.shape[1]), int(x.shape[2])))(x)
    # y = Conv2D(int(input_channel*radio),kernel_regularizer=regularizers.l2(l=l2),
    #            kernel_size=1,
    #            padding='same',
    #            use_bias=False)(y)
    # y = Activation('relu')(y)
    # y = Conv2D(input_channel,kernel_regularizer=regularizers.l2(l=l2),
    #            kernel_size=1,
    #            padding='same',
    #            use_bias=False)(y)
    # x = Multiply()([Activation('sigmoid')(y), x])

    # input_channel = K.int_shape(x)[-1]
    y = DepthwiseConv2D(kernel_regularizer=regularizers.l2(l=l2),
               kernel_size=3,
               padding='same',
               use_bias=False)(x)
    y = LeakyReLU(l_alpha)(y)
    y = DepthwiseConv2D(kernel_regularizer=regularizers.l2(l=l2),
               kernel_size=1,
               padding='same',
               use_bias=False)(y)
    x = Multiply()([Activation('sigmoid')(y), x])

    return x





def conv_block(x, filter, k_size=(3, 3)):
    x = Conv2D(filter, k_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def VGG16(input_shape, classes):
    img_input = Input(shape=input_shape)
    x = img_input
    # 第一个卷积部分
    # 112，112，64
    x = conv_block(x, base_filter, (3, 3))
    x = se_block(x)
    # x = conv_block(x, base_filter, (3, 3))
    # x = se_block(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # 第二个卷积部分
    # 56,56,128
    x = conv_block(x, base_filter*2, (3, 3))
    x = se_block(x)
    # x = conv_block(x, base_filter*2, (3, 3))
    # x = se_block(x)
    # x = conv_block(x, base_filter*2, (3, 3))
    # x = se_block(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # 第三个卷积部分
    # 28,28,256
    x = conv_block(x, base_filter*4, (3, 3))
    x = se_block(x)
    # x = conv_block(x, base_filter*4, (3, 3))
    # x = se_block(x)
    # x = conv_block(x, base_filter*4, (3, 3))
    # x = se_block(x)
    # x = conv_block(x, base_filter*4, (3, 3))
    # x = se_block(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # 第四个卷积部分
    # 14,14,512
    x = conv_block(x, base_filter*8, (3, 3))
    x = se_block(x)
    x = conv_block(x, base_filter*8, (3, 3))
    x = se_block(x)
    # x = conv_block(x, base_filter*8, (3, 3))
    # x = se_block(x)
    # x = conv_block(x, base_filter*8, (3, 3))
    # x = se_block(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # 第五个卷积部分
    # 7,7,512
    x = conv_block(x, base_filter*8, (3, 3))
    x = se_block(x)
    x = conv_block(x, base_filter*8, (3, 3))
    x = se_block(x)
    # x = conv_block(x, base_filter*8, (3, 3))
    # x = se_block(x)
    # x = conv_block(x, base_filter*8, (3, 3))
    # x = se_block(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # 提取特征

    # 分类部分
    # 7x7x512
    # 25088
    x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(img_input, x, name='vgg16')

    return model
