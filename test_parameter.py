from nets.UNet16 import UNet
print(UNet(n_classes=200, input_shape=(192, 192, 3), standard_input=False).summary())
# from nets.Mnasnet import MnasNet
# print(MnasNet(n_classes=10, input_shape=(224, 224, 3), standard_input=False).summary())
#
# from nets.Ghostnet_10 import GhostNet
# print(GhostNet(n_classes=200, inputs=(224, 224, 3), standard_input=False).summary())
# from keras_flops import get_flops
#
# flops = get_flops(UNet(n_classes=10, input_shape=(32, 32, 3), standard_input=True), batch_size=1)
# print(f"FLOPS: {flops / 10 ** 6:.03} M")
# print(f"FLOPS: {flops / 10 ** 3:.03} K")
# print("FLOPS: " + str(flops))
# from nets.mobilenetV2_10 import MobileNetV2
# print(MobileNetV2(input_shape=(32, 32, 3), standard_input=True, classes=200).summary())
# from nets.mobilenetV3_small_10 import MobileNetv3_small
# print(MobileNetv3_small(input_shape=(32, 32, 3), standard_input=True, classes=200).summary())

# from nets.shufflev1 import ShuffleNetV1
# print(ShuffleNetV1(n_classes=10, input_shape=(32, 32, 3), standard_input=True).summary())
# from kerascv.model_provider import get_model
# from nets.Shufflenetv2_10 import ShuffleNetV2
# print(ShuffleNetV2(classes=200).summary())