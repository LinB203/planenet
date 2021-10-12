from keras import layers, models, regularizers
from keras import backend as K
from keras.layers import Lambda, Activation


def MnasNet(n_classes=10, input_shape=(32, 32, 3), alpha=1.5, standard_input=True):

    inputs = layers.Input(shape=input_shape)
    if standard_input:
        x = Lambda(lambda img: K.resize_images(img, 7, 7, data_format='channels_last'), input_shape=input_shape)(inputs)
    else:
        x = inputs

    x = conv_bn(x, 32 * alpha, 3, strides=2)
    x = sepConv_bn_noskip(x, 16 * alpha, 3, strides=1)
    # MBConv3 3x3
    x = MBConv_idskip(x, filters=24, kernel_size=3, strides=2, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=24, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=24, kernel_size=3, strides=1, filters_multiplier=3, alpha=alpha)
    # MBConv3 5x5
    x = MBConv_idskip(x, filters=40, kernel_size=5, strides=2, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=40, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha)
    x = MBConv_idskip(x, filters=40, kernel_size=5, strides=1, filters_multiplier=3, alpha=alpha)
    # MBConv6 5x5
    x = MBConv_idskip(x, filters=80, kernel_size=5, strides=2, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=80, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=80, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 3x3
    x = MBConv_idskip(x, filters=96, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=96, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 5x5
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=2, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    x = MBConv_idskip(x, filters=192, kernel_size=5, strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 3x3
    x = MBConv_idskip(x, filters=320, kernel_size=3, strides=1, filters_multiplier=6, alpha=alpha)
    # FC + POOL
    x = conv_bn(x, filters=1152 * alpha, kernel_size=1, strides=1)
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=predictions)


def conv_bn(x, filters, kernel_size, strides=1, alpha=1, activation=True):
    filters = _make_divisible(filters * alpha)
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    if activation:
        x = Activation(relu6)(x)
    return x

def relu6(x):
    return K.relu(x, max_value=6)

def depthwiseConv_bn(x, depth_multiplier, kernel_size, strides=1):
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                               padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(relu6)(x)
    return x


def sepConv_bn_noskip(x, filters, kernel_size, strides=1):
    x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
    x = conv_bn(x, filters=filters, kernel_size=1, strides=1)
    return x


def MBConv_idskip(x_input, filters, kernel_size, strides=1, filters_multiplier=1, alpha=1.0):
    depthwise_conv_filters = _make_divisible(x_input.shape[3].value)
    pointwise_conv_filters = _make_divisible(filters * alpha)

    x = conv_bn(x_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1, strides=1)
    x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
    x = conv_bn(x, filters=pointwise_conv_filters, kernel_size=1, strides=1, activation=False)

    if strides == 1 and x.shape[3] == x_input.shape[3]:
        return layers.add([x_input, x])
    else:
        return x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
