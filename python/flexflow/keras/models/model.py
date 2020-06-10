import flexflow.core as ff

from .base_model import BaseModel
from .input_layer import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate

class Model(BaseModel):
  def __init__(self, input_tensors, output_tensor):
    super(Model, self).__init__()
    
    if (isinstance(input_tensors, list) == False):
       input_tensors = [input_tensors]
       
    self.input_tensors = input_tensors
    self.output_tensor = output_tensor
    
    self._init_dag()
    
  def _add_layer_metadata(self, layer):
    self._layers[self._nb_layers] = layer
    #assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
      
  def _init_dag(self):
    bfs_queue = []
    
    # for input_tensor in self.input_tensors:
    #   for layer in input_tensor.to_layers:
    #     bfs_queue.append(layer)
    # while(len(bfs_queue) != 0):
    #   layer = bfs_queue.pop(0)
    #   #print(layer)
    #   self._add_layer_metadata(layer)
    #   for child in layer.next_layers:
    #     if child not in bfs_queue:
    #       bfs_queue.append(child)
    #     else:
    #       print(child, "already in the queue")
    
    for input_tensor in reversed(self.input_tensors):
      for layer in reversed(input_tensor.to_layers):
        bfs_queue.append(layer)
    while(len(bfs_queue) != 0):
      layer = bfs_queue.pop()
      #print(layer)
      self._add_layer_metadata(layer)
      for child in reversed(layer.next_layers):
        assert child not in bfs_queue, "already in the stack"
        if child.nb_visited_prev_layers == len(child.prev_layers)-1:
          bfs_queue.append(child)
        else:
          child.nb_visited_prev_layers += 1
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      layer.nb_visited_prev_layers = 0          
      
  def compile(self, optimizer):
    self._create_input_and_label_tensors()
    use_api = 1
    if (use_api == 1):
      self._create_flexflow_layers()
    else:
      self._create_flexflow_layers_v2()
      self._init_inout()
    
    self._verify_output_tensors()
    self._verify_input_tensors()
    self._compile(optimizer)
    
  def fit(self, input_tensors, label_tensor, epochs=1):
    assert self.output_tensor.ffhandle != 0, "tensor is not init"
    if (isinstance(input_tensors, list) == False):
       input_tensors = [input_tensors]
    self._verify_tensors(input_tensors, label_tensor)
    self._create_data_loaders(input_tensors, label_tensor)
    self._set_optimizer()     
    self.ffmodel.init_layers()
    self._train(epochs)
    