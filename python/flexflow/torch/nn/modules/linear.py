from flexflow.core import *

from .module import Module

class Linear(Module):
  def __init__(self, in_features, out_features, bias=True):
    print("create Linear")
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features