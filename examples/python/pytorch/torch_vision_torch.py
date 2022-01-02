import torch.nn as nn
import torchvision.models as models
from flexflow.torch.model import PyTorchModel

# model = models.alexnet()

# model = models.vgg16()

# model = models.squeezenet1_0()

# model = models.densenet161()

# model = models.inception_v3()

model = models.googlenet()

# model = models.shufflenet_v2_x1_0()

# model = models.mobilenet_v2()
ff_torch_model = PyTorchModel(model)
ff_torch_model.torch_to_file("googlenet.ff")