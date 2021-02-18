import torch.nn as nn
import torchvision.models as models
import flexflow.torch.fx as fx

# alexnet = models.alexnet()
# fx.torch_to_flexflow(alexnet, "alexnet.ff")
#
# vgg16 = models.vgg16()
# fx.torch_to_flexflow(vgg16, "vgg16.ff")
#
# squeezenet = models.squeezenet1_0()
# fx.torch_to_flexflow(squeezenet, "squeezenet.ff")

# densenet = models.densenet161()
# fx.torch_to_flexflow(densenet, "densenet.ff")

# inception = models.inception_v3()
# fx.torch_to_flexflow(inception, "inception.ff")

googlenet = models.googlenet()
fx.torch_to_flexflow(googlenet, "googlenet.ff")

# shufflenet = models.shufflenet_v2_x1_0()
# fx.torch_to_flexflow(shufflenet, "shufflenet.ff")

# mobilenet = models.mobilenet_v2()
# fx.torch_to_flexflow(mobilenet, "mobilenet.ff")