from keras.models import Model
import keras
from keras.layers import *


def MobileNetV3(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                classifier_activation='softmax'):
    img_input = Input(shape=input_shape)
    channel_axis = -1
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

    x = img_input
    x = Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = Activation(activation)(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(K.int_shape(x)[channel_axis] * 6)

    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = Activation(activation)(x)
    x = Conv2D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2')(x)
    x = Activation(activation)(x)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_point_ch))(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
    x = Flatten()(x)
    x = Activation(activation=classifier_activation,
                   name='Predictions')(x)
    model = Model(img_input, x, name='MobilenetV3' + model_type)

    return model


def MobileNetV3Small(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
        return x

    return MobileNetV3(stack_fn, 1024, input_shape, alpha, 'small', minimalistic,
                       include_top, weights, input_tensor, classes, pooling,
                       dropout_rate, classifier_activation)


def MobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     classifier_activation='softmax'):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)
        return x

    return MobileNetV3(stack_fn, 1280, input_shape, alpha, 'large', minimalistic,
                       include_top, weights, input_tensor, classes, pooling,
                       dropout_rate, classifier_activation)


from keras import backend as K
def relu(x):
    return K.relu(x)

def hard_sigmoid(x):
    return K.relu(x + 3.0, max_value=6.0) / 6.0


def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    x = Reshape((1, 1, filters))(x)
    x = Conv2D(_depth(filters * se_ratio),
                kernel_size=1,
                padding='same',
                activation='relu',
                name=prefix + 'squeeze_excite/Conv')(x)
    x = Conv2D(filters,
                kernel_size=1,
                padding='same',
                name=prefix + 'squeeze_excite/Conv_1')(x)
    x = Activation(hard_sigmoid)(x)
    x = Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = K.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(x)
        x = BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(x)
        x = Activation(activation)(x)

    if stride == 2:
        x = ZeroPadding2D(
            padding=correct_pad(x, kernel_size),
            name=prefix + 'depthwise/pad')(x)
    x = DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(x)
    x = BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(x)
    x = Activation(activation)(x)
    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)
    x = Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(x)
    x = BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(x)
    if stride == 1 and infilters == filters:
        x = Add(name=prefix + 'Add')([shortcut, x])
    return x
