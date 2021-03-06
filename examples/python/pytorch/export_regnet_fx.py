import classy_vision.models.regnet as rgn
import flexflow.torch.fx as fx
import torch.nn as nn

model = rgn.RegNetX32gf()
model = nn.Sequential(model,nn.Flatten(),nn.Linear(2520*7*7,1000))
fx.torch_to_flexflow(model, "regnetX32gf.ff")
