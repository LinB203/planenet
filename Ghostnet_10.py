from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dropout

from keras import backend as K
from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Reshape, Multiply
from keras.layers import Lambda, Concatenate

import math


def slices(dw, n, data_format='channels_last'):
    if data_format == 'channels_last':
        return dw[:, :, :, :n]
    else:
        return dw[:, :n, :, :]


def _conv_block(inputs, outputs, kernel, strides, padding='same',
                use_relu=True, use_bias=False, data_format='channels_last'):
    channel_axis = -1 if K.image_data_format() == 'channels_last' else 1

    x = Conv2D(outputs, kernel, padding=padding, strides=strides, use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    if use_relu:
        x = Activation('relu')(x)

    return x


def _squeeze(inputs, exp, ratio, data_format='channels_last'):
    input_channels = int(inputs.shape[-1]) if K.image_data_format() == 'channels_last' else int(inputs.shape[1])

    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, input_channels))(x)

    x = Conv2D(math.ceil(exp / ratio), (1, 1), strides=(1, 1), padding='same',
               data_format=data_format, use_bias=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(exp, (1, 1), strides=(1, 1), padding='same',
               data_format=data_format, use_bias=False)(x)
    x = Activation('hard_sigmoid')(x)

    x = Multiply()([inputs, x])  # inputs和x逐元素相乘

    return x


def _ghost_module(inputs, exp, kernel, dw_kernel, ratio, s=1,
                  padding='SAME', use_bias=False, data_format='channels_last',
                  activation=None):
    output_channels = math.ceil(exp * 1.0 / ratio)

    x = Conv2D(output_channels, kernel, strides=(s, s), padding=padding,
               activation=activation, data_format=data_format,
               use_bias=use_bias)(inputs)

    if ratio == 1:
        return x

    dw = DepthwiseConv2D(dw_kernel, s, padding=padding, depth_multiplier=ratio - 1,
                         activation=activation,
                         use_bias=use_bias)(x)

    dw = Lambda(slices, arguments={'n': exp - output_channels, 'data_format': data_format})(dw)

    x = Concatenate(axis=-1 if data_format == 'channels_last' else 1)([x, dw])

    return x


def _ghost_bottleneck(inputs, outputs, kernel, dw_kernel,
                      exp, s, ratio, squeeze, name=None):
    data_format = K.image_data_format()
    channel_axis = -1 if data_format == 'channels_last' else 1

    input_shape = K.int_shape(inputs)  # 获取输入张量的尺寸

    if s == 1 and input_shape[channel_axis] == outputs:
        res = inputs
    else:
        res = DepthwiseConv2D(kernel, strides=s, padding='SAME', depth_multiplier=ratio - 1,
                              data_format=data_format, activation=None, use_bias=False)(inputs)
        res = BatchNormalization(axis=channel_axis)(res)
        res = _conv_block(res, outputs, (1, 1), (1, 1), padding='valid',
                          use_relu=False, use_bias=False, data_format=data_format)

    x = _ghost_module(inputs, exp, [1, 1], dw_kernel, ratio)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if s > 1:
        x = DepthwiseConv2D(dw_kernel, s, padding='same', depth_multiplier=ratio - 1,
                            data_format=data_format, activation=None, use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

    if squeeze:
        x = _squeeze(x, exp, 4, data_format=data_format)

    x = _ghost_module(x, outputs, [1, 1], dw_kernel, ratio)
    x = BatchNormalization(axis=channel_axis)(x)

    x = Add()([res, x])

    return x


def GhostNet(n_classes=10, inputs=(32, 32, 3), standard_input=True):
    ratio = 2
    dw_kernel = 3
    inputs = Input(shape=inputs)
    if standard_input:
        x = Lambda(lambda img: K.resize_images(img, 7, 7, data_format='channels_last'), input_shape=inputs)(inputs)
    else:
        x = inputs

    x = _conv_block(x, 16, (3, 3), strides=(2, 2))

    x = _ghost_bottleneck(x, 16, (3, 3), dw_kernel, 16, 1, ratio, False, name='ghost_bottleneck1')
    x = _ghost_bottleneck(x, 24, (3, 3), dw_kernel, 48, 2, ratio, False, name='ghost_bottleneck2')

    x = _ghost_bottleneck(x, 24, (3, 3), dw_kernel, 72, 1, ratio, False, name='ghost_bottleneck3')
    x = _ghost_bottleneck(x, 40, (5, 5), dw_kernel, 72, 2, ratio, True, name='ghost_bottleneck4')

    x = _ghost_bottleneck(x, 40, (5, 5), dw_kernel, 120, 1, ratio, True, name='ghost_bottleneck5')
    x = _ghost_bottleneck(x, 80, (3, 3), dw_kernel, 240, 2, ratio, False, name='ghost_bottleneck6')

    x = _ghost_bottleneck(x, 80, (3, 3), dw_kernel, 200, 1, ratio, False, name='ghost_bottleneck7')
    x = _ghost_bottleneck(x, 80, (3, 3), dw_kernel, 184, 1, ratio, False, name='ghost_bottleneck8')
    x = _ghost_bottleneck(x, 80, (3, 3), dw_kernel, 184, 1, ratio, False, name='ghost_bottleneck9')
    x = _ghost_bottleneck(x, 112, (3, 3), dw_kernel, 480, 1, ratio, True, name='ghost_bottleneck10')
    x = _ghost_bottleneck(x, 112, (3, 3), dw_kernel, 672, 1, ratio, True, name='ghost_bottleneck11')
    x = _ghost_bottleneck(x, 160, (5, 5), dw_kernel, 672, 2, ratio, True, name='ghost_bottleneck12')

    x = _ghost_bottleneck(x, 160, (5, 5), dw_kernel, 960, 1, ratio, False, name='ghost_bottleneck13')
    x = _ghost_bottleneck(x, 160, (5, 5), dw_kernel, 960, 1, ratio, True, name='ghost_bottleneck14')
    x = _ghost_bottleneck(x, 160, (5, 5), dw_kernel, 960, 1, ratio, False, name='ghost_bottleneck15')
    x = _ghost_bottleneck(x, 160, (5, 5), dw_kernel, 960, 1, ratio, True, name='ghost_bottleneck16')

    x = _conv_block(x, 960, (1, 1), strides=1)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 960))(x)

    x = _conv_block(x, 1280, (1, 1), strides=1)

    x = Dropout(rate=0.05)(x)
    x = Conv2D(n_classes, (1, 1), strides=1, padding='same',
               data_format='channels_last', name='last_Conv',
               activation='softmax', use_bias=False)(x)

    x = Reshape((n_classes,))(x)
    model = Model(inputs, x)
    return model
