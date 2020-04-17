from flexflow.core import *

from .op import Op

class Linear(Op):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__()
    print("create Linear")
    self.in_features = in_features
    self.out_features = out_features
    self.handle = 0
    
  def __call__(self, input):
    return self.forward(input)
      
  def forward(self, input):
    print("linear forward ", self.layer_id);
    self.handle.forward(self._ffmodel)
    return input