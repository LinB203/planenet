from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Input
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape, Lambda, AvgPool2D, Dropout, Flatten, \
    Softmax
from keras.models import Model
from keras import backend as K

alpha = 0.5


def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)


def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def _activation(x, name='relu'):
    if name == 'relu':
        return Activation(relu6)(x)
    elif name == 'hardswish':
        return Activation(hard_swish)(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetv3_large(input_shape, classes, standard_input, dropout=0.2):
    regularizer = None

    img_input = Input(shape=input_shape)
    if standard_input:
        x = Lambda(lambda img: K.resize_images(img, 7, 7, data_format='channels_last'), input_shape=input_shape)(
            img_input)
    else:
        x = img_input

    channel_axis = -1

    kernel = 5
    activation = 'hardswish'
    se_ratio = 0.25

    x = Conv2D(16,
               kernel_size=3,
               strides=(2, 2),
               padding='same',
               kernel_regularizer=regularizer,
               use_bias=False,
               name='Conv')(x)
    x = BatchNormalization(axis=channel_axis,
                           name='Conv/BatchNorm')(x)
    x = _activation(x, name=activation)

    # if model_type == 'small':
    #     #   (inputs, expansion, alpha, out_ch, kernel_size, stride, se_ratio, activation, regularizer, block_id)
    #     x = _inverted_res_block(x, 1, 16, alpha, 3, 2, se_ratio, 'relu', regularizer, 0)
    #     x = _inverted_res_block(x, 4.5, 24, alpha, 3, 2, None, 'relu', regularizer, 1)
    #     x = _inverted_res_block(x, 3.66, 24, alpha, 3, 1, None, 'relu', regularizer, 2)
    #     x = _inverted_res_block(x, 4, 40, alpha, kernel, 2, se_ratio, activation, regularizer, 3)
    #     x = _inverted_res_block(x, 6, 40, alpha, kernel, 1, se_ratio, activation, regularizer, 4)
    #     x = _inverted_res_block(x, 6, 40, alpha, kernel, 1, se_ratio, activation, regularizer, 5)
    #     x = _inverted_res_block(x, 3, 48, alpha, kernel, 1, se_ratio, activation, regularizer, 6)
    #     x = _inverted_res_block(x, 3, 48, alpha, kernel, 1, se_ratio, activation, regularizer, 7)
    #     x = _inverted_res_block(x, 6, 96, alpha, kernel, 2, se_ratio, activation, regularizer, 8)
    #     x = _inverted_res_block(x, 6, 96, alpha, kernel, 1, se_ratio, activation, regularizer, 9)
    #     x = _inverted_res_block(x, 6, 96, alpha, kernel, 1, se_ratio, activation, regularizer, 10)
    #     last_conv_ch = _make_divisible(576 * alpha, 8)
    #     last_point_ch = 1024
    x = _inverted_res_block(x, 1, 16, alpha, 3, 1, None, 'relu', regularizer, 0)
    x = _inverted_res_block(x, 4, 24, alpha, 3, 2, None, 'relu', regularizer, 1)
    x = _inverted_res_block(x, 3, 24, alpha, 3, 1, None, 'relu', regularizer, 2)
    x = _inverted_res_block(x, 3, 40, alpha, kernel, 2, se_ratio, 'relu', regularizer, 3)
    x = _inverted_res_block(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', regularizer, 4)
    x = _inverted_res_block(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', regularizer, 5)
    x = _inverted_res_block(x, 6, 80, alpha, 3, 2, None, activation, regularizer, 6)
    x = _inverted_res_block(x, 2.5, 80, alpha, 3, 1, None, activation, regularizer, 7)
    x = _inverted_res_block(x, 2.3, 80, alpha, 3, 1, None, activation, regularizer, 8)
    x = _inverted_res_block(x, 2.3, 80, alpha, 3, 1, None, activation, regularizer, 9)
    x = _inverted_res_block(x, 6, 112, alpha, 3, 1, se_ratio, activation, regularizer, 10)
    x = _inverted_res_block(x, 6, 112, alpha, 3, 1, se_ratio, activation, regularizer, 11)
    x = _inverted_res_block(x, 6, 160, alpha, kernel, 2, se_ratio, activation, regularizer, 12)
    x = _inverted_res_block(x, 6, 160, alpha, kernel, 1, se_ratio, activation, regularizer, 13)
    x = _inverted_res_block(x, 6, 160, alpha, kernel, 1, se_ratio, activation, regularizer, 14)
    last_conv_ch = _make_divisible(960 * alpha, 8)
    last_point_ch = 1280

    if alpha > 1.0:
        last_point_ch = _make_divisible(last_point_ch * alpha, 8)

    x = Conv2D(last_conv_ch,
               kernel_size=1,
               strides=1,
               padding='same',
               kernel_regularizer=regularizer,
               use_bias=False,
               name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis,
                           name='Conv_1/BatchNorm')(x)
    x = _activation(x, name=activation)

    # AvgPool2D is optimized in TFLite unlike GlobalAveragePooling2D
    x = AvgPool2D(pool_size=(int(x.shape[1]), int(x.shape[2])))(x)
    x = Conv2D(last_point_ch,
               kernel_size=1,
               padding='same',
               kernel_regularizer=regularizer,
               bias_regularizer=regularizer,
               name='Conv_2')(x)
    x = _activation(x, name=activation)
    x = Dropout(dropout)(x)

    x = Conv2D(classes,
               kernel_size=1,
               padding='same',
               use_bias=True,
               kernel_regularizer=regularizer,
               bias_regularizer=regularizer,
               name='Logits')(x)
    x = Flatten()(x)
    x = Softmax(name='Predictions/Softmax')(x)

    # Create model.
    model = Model(img_input, x, name='MobilenetV3')

    return model


def _inverted_res_block(inputs, expansion, alpha, out_ch, kernel_size, stride, se_ratio, activation, regularizer,
                        block_id):
    channel_axis = -1
    in_channels = K.int_shape(inputs)[channel_axis]
    out_channels = _make_divisible(out_ch * alpha, 8)
    exp_size = _make_divisible(in_channels * expansion, 8)
    x = inputs
    prefix = 'expanded_conv/'
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(exp_size,
                   kernel_size=1,
                   padding='same',
                   kernel_regularizer=regularizer,
                   use_bias=False,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                               name=prefix + 'expand/BatchNorm')(x)
        x = _activation(x, activation)

    x = DepthwiseConv2D(kernel_size,
                        strides=stride,
                        padding='same',
                        dilation_rate=1,
                        depthwise_regularizer=regularizer,
                        use_bias=False,
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                           name=prefix + 'depthwise/BatchNorm')(x)
    x = _activation(x, activation)

    if se_ratio:
        reduced_ch = _make_divisible(exp_size * se_ratio, 8)
        y = AvgPool2D(pool_size=(int(x.shape[1]), int(x.shape[2])),
                      name=prefix + 'squeeze_excite/AvgPool')(x)
        y = Conv2D(reduced_ch,
                   kernel_size=1,
                   padding='same',
                   kernel_regularizer=regularizer,
                   use_bias=True,
                   name=prefix + 'squeeze_excite/Conv')(y)
        y = Activation('relu')(y)
        y = Conv2D(exp_size,
                   kernel_size=1,
                   padding='same',
                   kernel_regularizer=regularizer,
                   use_bias=True,
                   name=prefix + 'squeeze_excite/Conv_1')(y)
        x = Multiply(name=prefix + 'squeeze_excite/Mul')([Activation(hard_swish)(y), x])

    x = Conv2D(out_channels,
               kernel_size=1,
               padding='same',
               kernel_regularizer=regularizer,
               use_bias=False,
               name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                           name=prefix + 'project/BatchNorm')(x)

    if in_channels == out_channels and stride == 1:
        x = Add(name=prefix + 'Add')([inputs, x])

    return x
