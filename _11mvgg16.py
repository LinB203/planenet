from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, \
    DepthwiseConv2D, Reshape, GlobalAveragePooling2D, Add, Activation, \
    BatchNormalization, Flatten, Dropout, Input, Multiply, Lambda, Concatenate, LeakyReLU
from keras.models import Model
from keras import backend as K
from tensorflow.keras import backend

def _inverted_res_block(inputs, expansion, stride, alpha, filters):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    x = inputs
    x = DepthwiseConv2D(kernel_size=1,
                        strides=stride,
                        activation=None,
                        use_bias=False,
                        padding='same',
                        depth_multiplier=expansion)(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.99)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(0.5)(x)

    x = Conv2D(pointwise_conv_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None)(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.99)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(0.5)(x)

    x = DepthwiseConv2D(kernel_size=1,
                        strides=1,
                        activation=None,
                        use_bias=False,
                        padding='same')(x)
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.99)(x)
    # x = Activation(relu6)(x)
    x = LeakyReLU(0.5)(x)

    if in_channels == pointwise_conv_filters and stride == 1:
        x = Add()([inputs, x])
        return BatchNormalization(epsilon=1e-3,
                           momentum=0.99)(x)
    return x

def VGG16(input_shape, classes, standard_input, alpha=1, expansion=6):
    img_input = Input(shape=input_shape)
    x = img_input
    # 第一个卷积部分
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=2,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=2,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=2,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                        expansion=expansion)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=2,
                        expansion=expansion)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(img_input, x, name='mvgg16')

    return model
