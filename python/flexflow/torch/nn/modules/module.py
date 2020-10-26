import torch.nn as nn

import flexflow.torch.fx as fx

class Module(nn.Module):
  def __init__(self):
    super(Module, self).__init__()
    self._ffmodel = None
    self._graph = None
  
  def __call__(self, input):
    print("forward");
  
  # TODO: automatically call this function  
  def symbolic_trace(self):
    self._graph = fx.symbolic_trace(self)
    for node in self._graph:
      if type(node) == fx.ModuleNode:
        print(node.name, node.module)