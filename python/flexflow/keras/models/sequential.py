import flexflow.core as ff

from .base_model import BaseModel
from flexflow.keras.layers.base_layer import Layer
from .input_layer import Tensor, Input
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate

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
  
  def pop(self):
    assert 0, "Not implemented"
    
  def __add_layer(self, layer):
    self._layers.append(layer)
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    if layer.layer_id == 0:
      input_tensor = Input(batch_shape=layer.input_shape, dtype="float32")
      self._input_tensors.append(input_tensor)
      self._output_tensor = input_tensor
      
    self._output_tensor = layer(self._output_tensor)
    
    layer.verify_meta_data()
    
  def __add_model(self, model):
    for layer in model.layers:
      layer.reset_connection()
      self.__add_layer(layer)