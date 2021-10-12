import tensorflow as tf
import keras.backend as K
from nets.VGG16 import VGG16
from nets.splitnet import split_VGG16
from nets.res18 import resnet_v2, resnet_v1
# from nets.Ghostnet_10 import GhostNet
# from nets.Mnasnet_05 import MnasNet
# from nets.Mnasnet_15 import MnasNet
# from nets.Mnasnet_125 import MnasNet
# from nets.Mnasnet_10 import MnasNet
from nets.mobilenetV1_10 import MobileNetV1
# from nets.mobilenetV1_15 import MobileNetV1
# from nets.mobilenetV1_05 import MobileNetV1
# from nets.mobilenetV1_125 import MobileNetV1
# from nets.mobilenetV2_05 import MobileNetV2
# from nets.mobilenetV2_15 import MobileNetV2
from nets.mobilenetV2_10 import MobileNetV2
# from nets.mobilenetV2_125 import MobileNetV2
# from nets.mobilenetV3_large_05 import MobileNetv3_large
# from nets.mobilenetV3_large_10 import MobileNetv3_large
# from nets.mobilenetV3_large_15 import MobileNetv3_large
# from nets.mobilenetV3_large_125 import MobileNetv3_large
# from nets.mobilenetV3_small_10 import MobileNetv3_small
from nets.our import MobileNetV2
# from nets.mobilenetV3_small_05 import MobileNetv3_small
# from nets.mobilenetV3_small_125 import MobileNetv3_small
# from nets.mobilenetV3_small_15 import MobileNetv3_small
from nets.Shufflenetv1 import ShuffleNetV1
# from nets.Shufflenetv2_05 import ShuffleNetV2
from nets.Shufflenetv2_10 import ShuffleNetV2
# from nets.Shufflenetv2_20 import ShuffleNetV2
# from nets.Shufflenetv2_15 import ShuffleNetV2
from nets.Squeezenet import SqueezeNet
# from nets.UNet21 import UNet
# model = GhostNet(n_classes=200, inputs=(192, 192, 3), standard_input=False)
# model = MnasNet(n_classes=200, input_shape=(192, 192, 3), standard_input=False)
# model = MobileNetV1(classes=200, input_shape=(192, 192, 3), standard_input=False)
# model = MobileNetV2(classes=200, input_shape=(192, 192, 3), standard_input=False)
# model = MobileNetV2(classes=10, input_shape=(224, 224, 3),alpha=0.5)
# model = MobileNetv3_large(classes=200, input_shape=(192, 192, 3), standard_input=False)
# model = ShuffleNetV1(classes=1000, input_shape=(32, 32, 3), standard_input=False)
model = ShuffleNetV2(classes=1000, input_shape=(224, 224, 3), standard_input=False)
# model = SqueezeNet(n_classes=200, inputs=(192, 192, 3), standard_input=False)
# model = UNet(n_classes=10, input_shape=(256, 256, 3))
# model = VGG16(input_shape=(32, 32, 3), classes=10)
# model = split_VGG16(input_shape=(32, 32, 3), classes=10)



# n = 18
# version = 1
# if version == 1:
#     depth = n * 6 + 2
# elif version == 2:
#     depth = n * 9 + 2
# if version == 2:
#     model = resnet_v2(input_shape=(32, 32, 3), depth=depth)
# else:
#     model = resnet_v1(input_shape=(32, 32, 3), depth=depth)
def cal_flops_params(model):
    sum_flops = 0
    sum_params = 0
    summary = model.summary()
    for layer in summary:
        if layer[2]:
            if 'Conv2D' in layer[0]:
                sum_params += layer[2]
                sum_flops += layer[1][1]*layer[1][2]*layer[2]
            if 'Dense' in layer[0]:
                sum_params += layer[2]
                sum_flops += layer[1][1]

    print('sum_flops:{:.2f}M, sum_params:{:.2f}M'.format(sum_flops/1000000, sum_params/1000000))
    return sum_flops, sum_params

cal_flops_params(model)
