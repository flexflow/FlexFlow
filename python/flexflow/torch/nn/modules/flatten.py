from flexflow.core import *

from .op import Op

class Flatten(Op):
  def __init__(self, start_dim=1, end_dim=-1):
    super(Flatten, self).__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim
    
  def __call__(self, input):
    return self.forward(input)
    
  def forward(self, input):
    print("flat forward ", self.layer_id);
    self.handle.forward(self._ffmodel)
    return input
