# import math
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import backend
# from keras import backend as K
# from keras.preprocessing import image
# from keras.models import Model
# from keras.layers.normalization import BatchNormalization
# from keras.layers import Conv2D, Add, ZeroPadding2D, GlobalAveragePooling2D, Dropout, Dense, Lambda, Multiply, \
#     LeakyReLU, Reshape, regularizers
# from keras.layers import MaxPooling2D, Activation, DepthwiseConv2D, Input, GlobalMaxPooling2D
#
#
# def relu6(x):
#     return K.relu(x, max_value=6)
#
#
# def relu(x):
#     return K.relu(x)
#
#
# def hard_sigmoid(x):
#     return K.relu(x + 3.0, max_value=6.0) / 6.0
#
#
# def hard_swish(x):
#     return x * K.relu(x + 3.0, max_value=6.0) / 6.0
#
#
# l_alpha = 0.5
#
#
# def correct_pad(inputs, kernel_size):
#     img_dim = 1
#     input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
#
#     if isinstance(kernel_size, int):
#         kernel_size = (kernel_size, kernel_size)
#
#     if input_size[0] is None:
#         adjust = (1, 1)
#     else:
#         adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
#
#     correct = (kernel_size[0] // 2, kernel_size[1] // 2)
#
#     return ((correct[0] - adjust[0], correct[0]),
#             (correct[1] - adjust[1], correct[1]))
#
#
# def _se_block(inputs, filters, se_ratio):
#     x = GlobalAveragePooling2D()(inputs)
#     x = Reshape((1, 1, filters))(x)
#     x = Conv2D(int(filters * se_ratio),
#                kernel_size=1,
#                padding='same',
#                activation='relu')(x)
#     x = Conv2D(filters,
#                kernel_size=1,
#                padding='same')(x)
#     x = Activation(hard_sigmoid)(x)
#     x = Multiply()([inputs, x])
#     return x
#
#
# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# def MobileNetV2(input_shape, classes, alpha=1.0, expansion=6):
#     img_input = Input(shape=input_shape)
#     x = img_input
#     x = _inverted_res_block(x, filters=16, alpha=alpha, stride=2, expansion=6, se_ratio=None)
#     x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=6, se_ratio=None)
#
#     x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=5, se_ratio=None)
#     x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=5, se_ratio=None)
#
#     x = _inverted_res_block(x, filters=40, alpha=alpha, stride=2, expansion=5, se_ratio=None)
#     x = _inverted_res_block(x, filters=40, alpha=alpha, stride=1, expansion=5, se_ratio=None)
#     x = _inverted_res_block(x, filters=40, alpha=alpha, stride=1, expansion=5, se_ratio=None)
#
#     x = _inverted_res_block(x, filters=48, alpha=alpha, stride=2, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=48, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=48, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#     x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=3, se_ratio=None)
#
#     if alpha > 1.0:
#         last_block_filters = _make_divisible(880 * alpha, 8)
#     else:
#         last_block_filters = 880
#
#     # 7,7,320 -> 7,7,1280
#     x = Conv2D(last_block_filters, kernel_size=1,  kernel_regularizer=regularizers.l2(l=0.0003), use_bias=False)(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
#     # x = Activation(relu6)(x)
#     x = LeakyReLU(l_alpha)(x)
#
#     # 7,7,1280 -> 1,1,1280
#     x = GlobalAveragePooling2D()(x)
#
#     x = Dense(classes, activation='softmax', use_bias=True)(x)
#
#     inputs = img_input
#
#     model = Model(inputs, x)
#
#     return model
#
#
# def _inverted_res_block(inputs, expansion, stride, alpha, filters, se_ratio):
#     in_channels = backend.int_shape(inputs)[-1]
#     pointwise_conv_filters = int(filters * alpha)
#     x = inputs
#     x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, kernel_regularizer=regularizers.l2(l=0.0003),
#                         use_bias=False, padding='same', depth_multiplier=expansion)(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
#     # x = Activation(relu6)(x)
#     x = LeakyReLU(l_alpha)(x)
#     if se_ratio:
#         x = _se_block(x, int(backend.int_shape(x)[-1]), se_ratio)
#
#     x = Conv2D(pointwise_conv_filters, kernel_regularizer=regularizers.l2(l=0.0003),
#                kernel_size=1,
#                padding='same',
#                use_bias=False,
#                activation=None)(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
#     # x = Activation(relu6)(x)
#     x = LeakyReLU(l_alpha)(x)
#
#     x = DepthwiseConv2D(kernel_size=3, kernel_regularizer=regularizers.l2(l=0.0003),
#                         strides=1,
#                         activation=None,
#                         use_bias=False,
#                         padding='same')(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
#     # x = Activation(relu6)(x)
#     x = LeakyReLU(l_alpha)(x)
#
#     if in_channels == pointwise_conv_filters and stride == 1:
#         x = Add()([inputs, x])
#         return BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
#     return x
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Add, ZeroPadding2D, GlobalAveragePooling2D, Dropout, Dense, Lambda, Multiply, \
    LeakyReLU, Reshape, regularizers
from keras.layers import MaxPooling2D, Activation, DepthwiseConv2D, Input, GlobalMaxPooling2D


def relu6(x):
    return K.relu(x, max_value=6)


def relu(x):
    return K.relu(x)


def hard_sigmoid(x):
    return K.relu(x + 3.0, max_value=6.0) / 6.0


def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


l_alpha = 0.5




def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(input_shape, classes, alpha=1.0, expansion=6):
    img_input = Input(shape=input_shape)
    x = img_input
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=2, expansion=6, se_ratio=None)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=6, se_ratio=None)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=5, se_ratio=None)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=5, se_ratio=None)
    x = _inverted_res_block(x, filters=40, alpha=alpha, stride=2, expansion=5, se_ratio=None)

    x = _inverted_res_block(x, filters=40, alpha=alpha, stride=1, expansion=5, se_ratio=None)
    x = _inverted_res_block(x, filters=40, alpha=alpha, stride=1, expansion=5, se_ratio=None)
    x = _inverted_res_block(x, filters=48, alpha=alpha, stride=2, expansion=3, se_ratio=None)

    x = _inverted_res_block(x, filters=48, alpha=alpha, stride=1, expansion=3, se_ratio=None)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=3, se_ratio=None)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=3, se_ratio=None)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=3, se_ratio=None)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=3, se_ratio=None)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    # 7,7,320 -> 7,7,1280
    x = Conv2D(last_block_filters, kernel_size=1,  kernel_regularizer=regularizers.l2(l=0.01), use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(l_alpha)(x)

    # 7,7,1280 -> 1,1,1280
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes, activation='softmax', use_bias=True)(x)

    inputs = img_input

    model = Model(inputs, x)

    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, se_ratio):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    x = inputs
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, kernel_regularizer=regularizers.l2(l=0.01),
                        use_bias=False, padding='same', depth_multiplier=expansion)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(l_alpha)(x)
    x = Conv2D(pointwise_conv_filters, kernel_regularizer=regularizers.l2(l=0.01),
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(l_alpha)(x)

    x = DepthwiseConv2D(kernel_size=3, kernel_regularizer=regularizers.l2(l=0.01),
                        strides=1,
                        activation=None,
                        use_bias=False,
                        padding='same')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(l_alpha)(x)

    if in_channels == pointwise_conv_filters and stride == 1:
        x = Add()([inputs, x])
        return BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
    return x
