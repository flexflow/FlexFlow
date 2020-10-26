import torch.nn as nn

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__(in_features, out_features, bias)