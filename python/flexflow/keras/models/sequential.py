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

from .base_model import BaseModel
from flexflow.keras.layers.base_layer import Layer
from .tensor import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate, Input

class Sequential(BaseModel):
  def __init__(self, layers=None, name=None):
    super(Sequential, self).__init__(name) 
    
    if isinstance(layers, list):
      for layer in layers:
        self.add(layer)
  
  def add(self, item):
    if isinstance(item, Layer):
      assert item.layer_id == -1, "layer id is inited"
      self.__add_layer(item)
    elif isinstance(item, BaseModel):
      self.__add_model(item)
    elif isinstance(item, Tensor):
      self.__add_input(item)
  
  def pop(self):
    assert 0, "Not implemented"
    
  def __add_layer(self, layer):
    self._layers.append(layer)
    assert layer.ffhandle == None, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    if layer.layer_id == 0 and len(self._input_layers) == 0:
      assert layer.input_shape != None, "input shape is not set"
      input_tensor = Input(batch_shape=layer.input_shape, dtype="float32")
      self.__add_input(input_tensor)
      
    self._output_tensor = layer(self._output_tensor)
    
    layer.verify_meta_data()
    
  def __add_model(self, model):
    for layer in model.layers:
      layer.reset_connection()
      self.__add_layer(layer)
      
  def __add_input(self, tensor):
    self._input_tensors.append(tensor)
    self._output_tensor = tensor
    self._input_layers.append(tensor.from_layer)