from flexflow.core import *

from .op import Op

class Flatten(Op):
  def __init__(self, start_dim=1, end_dim=-1):
    super(Flatten, self).__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim
    self.handle = 0
    
  def __call__(self, input):
    return self.forward(input)
    
  def forward(self, input):
    input_tensor = input[0]
    ffmodel = input[1]
    print("flat forward ", self._layer_id);
    output_tensor = self.handle.init_input(ffmodel, input_tensor);
    return [output_tensor, ffmodel]
