import classy_vision.models.regnet as rgn
from flexflow.torch.model import PyTorchModel
import torch.nn as nn

model = rgn.RegNetX32gf()
model = nn.Sequential(model,nn.Flatten(),nn.Linear(2520*7*7,1000))
ff_torch_model = PyTorchModel(model)
ff_torch_model.torch_to_file("regnetX32gf.ff")
