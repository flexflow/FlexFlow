# Copyright 2020 Stanford University, Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger

from .base_model import BaseModel
from .tensor import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate, Input, InputLayer

class Model(BaseModel):
  def __init__(self, inputs, outputs, name=None):
    super(Model, self).__init__(name)
    
    if (isinstance(inputs, list) == False):
       inputs = [inputs]
    
    self._input_tensors = inputs
    for input_tensor in inputs:
      self._input_layers.append(input_tensor.from_layer)
    self._output_tensor = outputs
    
    self.__traverse_dag_dfs()
    fflogger.debug("nb_layers %d" %(self._nb_layers))
    
  def __call__(self, input_tensor):
    for layer in self._layers:
      layer.reset_layer()
    self._output_tensor = input_tensor
    for layer in self._layers:
      self._output_tensor = layer(self._output_tensor)
    self._input_tensors = [input_tensor]
    return self._output_tensor
    
  def _add_layer_metadata(self, layer):
    self._layers.append(layer)
    #assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == None, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1       

  def __traverse_dag_bfs(self):
    bfs_queue = []
    for input_layer in self._input_layers:
      bfs_queue.append(input_layer)
    while(len(bfs_queue) != 0):
      layer = bfs_queue.pop(0)
      if (isinstance(layer, InputLayer) == False):
       #fflogger.debug(layer)
        self._add_layer_metadata(layer)
      for child in layer.next_layers:
        assert child not in bfs_queue, "already in the stack"
        if child.nb_visited_prev_layers == len(child.prev_layers)-1:
          if child.has_visited == False:
            child.has_visited = True
            bfs_queue.append(child)
        else:
          child.nb_visited_prev_layers += 1
    for layer in self._layers:
      layer.nb_visited_prev_layers = 0
      layer.has_visited = False
    
  def __traverse_dag_dfs(self):    
    dfs_stack = []
    for input_layer in reversed(self._input_layers):
      dfs_stack.append(input_layer)
    while(len(dfs_stack) != 0):
      layer = dfs_stack.pop()
      if (isinstance(layer, InputLayer) == False):
        #fflogger.debug(layer)
        self._add_layer_metadata(layer)
      for child in reversed(layer.next_layers):
        assert child not in dfs_stack, "already in the stack"
        if child.nb_visited_prev_layers == len(child.prev_layers)-1:
          if child.has_visited == False:
            child.has_visited = True
            dfs_stack.append(child)
        else:
          child.nb_visited_prev_layers += 1
    for layer in self._layers:
      layer.nb_visited_prev_layers = 0
      layer.has_visited = False
