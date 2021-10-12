from keras.models import *
from keras.activations import relu
from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import Input

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def relu6(x):
    return relu(x, max_value=6)

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1].value  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)
    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    return x


def mobilenetV2(input_height=224, input_width=224, alpha=1):
    inputs = Input(shape=(input_height, input_width, 3))

    first_block_filters = _make_divisible(32 * alpha, 8)
    # 416,416 -> 208,208
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(inputs)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)
    f1 = x

    # 208,208 -> 104,104
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)
    f2 = x
    # 104,104 -> 52,52
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)
    f3 = x
    # ---------------------------------------------------------------#
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=2, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)
    f4 = x
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)
    f5 = x
    return inputs, [f1, f2, f3, f4, f5]


from keras.models import *
from keras.layers import *
import tensorflow as tf

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def resize_images(args):
    x = args[0]
    y = args[1]
    return tf.image.resize_images(x, (K.int_shape(y)[1], K.int_shape(y)[2]), align_corners=True)


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=224, input_width=224):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels
    # print('f4需要的shape:', f4.shape)
    o = f4
    # 26,26,512
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 52,52,512
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    # 52,52,768
    # print('f3需要的shape:',o.shape)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # 52,52,256
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 104,104,256
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    # 104,104,384
    # print('f2需要的shape:',o.shape)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    # 104,104,128
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    # 208,208,128
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    # print('f1需要的shape:',o.shape)
    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    x = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(img_input, x)
    return model


def mobilenetv2_unet(n_classes, input_shape=(32, 32, 3)):
    model = _unet(n_classes, mobilenetV2, input_height=input_shape[0], input_width=input_shape[1])
    model.model_name = "mobilenet_unet"
    return model
