from flexflow.core import *

from .op import Op

class Linear(Op):
  def __init__(self, in_features, out_features, bias=True):
    print("create Linear")
    self.in_features = in_features
    self.out_features = out_features
    
  def __call__(self, input):
      return self.forward(input)
      
  def forward(self, input):
      print("linear forward ", self._layer_id);
      return input+1