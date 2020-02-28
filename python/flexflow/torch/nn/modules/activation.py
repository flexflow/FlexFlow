from flexflow.core import *

from .module import Module

class ReLU(Module):
  def __init__(self, inplace=False):
    super(ReLU, self).__init__()
    self.inplace = inplace