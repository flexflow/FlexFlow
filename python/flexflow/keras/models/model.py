import flexflow.core as ff

from .base_model import BaseModel
from .input_layer import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate

class Model(BaseModel):
  def __init__(self, inputs, outputs):
    super(Model, self).__init__()
    
    if (isinstance(inputs, list) == False):
       inputs = [inputs]
       
    self._input_tensors = inputs
    self._output_tensor = outputs
    
    self.__traverse_dag_dfs()
    print("nb_layers", self._nb_layers)
    
  def __call__(self, input_tensor):
    for layer in self.layers:
      layer.reset_layer()
    self._output_tensor = input_tensor
    for layer in self.layers:
      self._output_tensor = layer(self._output_tensor)
    self._input_tensors = [input_tensor]
    return self._output_tensor
    
  def _add_layer_metadata(self, layer):
    self._layers.append(layer)
    #assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1       

  def __traverse_dag_bfs(self):
    bfs_queue = []
    for input_tensor in self._input_tensors:
     for layer in input_tensor.to_layers:
       bfs_queue.append(layer)
    while(len(bfs_queue) != 0):
     layer = bfs_queue.pop(0)
     #print(layer)
     self._add_layer_metadata(layer)
     for child in layer.next_layers:
       if child not in bfs_queue:
         bfs_queue.append(child)
       else:
         print(child, "already in the queue")
    
  def __traverse_dag_dfs(self):    
    dfs_stack = []
    for input_tensor in reversed(self._input_tensors):
      for layer in reversed(input_tensor.to_layers):
        dfs_stack.append(layer)
    while(len(dfs_stack) != 0):
      layer = dfs_stack.pop()
      #print(layer)
      self._add_layer_metadata(layer)
      for child in reversed(layer.next_layers):
        assert child not in dfs_stack, "already in the stack"
        if child.nb_visited_prev_layers == len(child.prev_layers)-1:
          dfs_stack.append(child)
        else:
          child.nb_visited_prev_layers += 1
    for layer in self._layers:
      layer.nb_visited_prev_layers = 0