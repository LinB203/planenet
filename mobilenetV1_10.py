import warnings
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, Lambda, Conv2D, Multiply
from keras import backend as K


def MobileNetV1(input_shape, classes, standard_input, depth_multiplier=1, dropout=1e-3, alpha=1):
    img_input = Input(shape=input_shape)
    if standard_input:
        x = Lambda(lambda img: K.resize_images(img, 7, 7, data_format='channels_last'), input_shape=input_shape)(
            img_input)
    else:
        x = img_input
    # 224,224,3 -> 112,112,32
    x = _conv_block(x, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1, alpha=alpha)

    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2, alpha=alpha)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3, alpha=alpha)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4, alpha=alpha)
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5, alpha=alpha)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6, alpha=alpha)
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7, alpha=alpha)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8, alpha=alpha)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9, alpha=alpha)
    # x = Multiply()([x, x])
    # x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10, alpha=alpha)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11, alpha=alpha)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12, alpha=alpha)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13, alpha=alpha)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, int(alpha * 1024)), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)

    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    # print(x.shape)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1')

    return model


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1, alpha=1.0):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)

