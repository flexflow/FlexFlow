import flexflow.core as ff

from .base_model import BaseModel
from flexflow.keras.layers.base_layer import Layer
from .input_layer import Tensor, Input
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate

class Sequential(BaseModel):
  def __init__(self, layer_list=[]):
    super(Sequential, self).__init__() 
    
    if len(layer_list) > 0:
      for layer in layer_list:
        self.add(layer)
  
  def add(self, item):
    if (isinstance(item, Layer)):
      assert item.layer_id == -1, "layer id is inited"
      self.__add_layer(item)
    elif (isinstance(item, BaseModel)):
      self.__add_model(item)
    
  def __add_layer(self, layer):
    self._layers[self._nb_layers] = layer
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    if (layer.layer_id == 0):
      input_tensor = Input(batch_shape=layer.input_shape, dtype="float32")
      self._input_tensors.append(input_tensor)
      self._output_tensor = input_tensor
      
    self._output_tensor = layer(self._output_tensor)
    
    layer.verify_meta_data()
    
  def compile(self, optimizer):
    self._create_input_and_label_tensors()
    self._create_flexflow_layers()
    #self._init_inout() 
    
    self._verify_output_tensors()
    self._verify_input_tensors()
    self._compile(optimizer)
    print(self._input_tensors[0], self._output_tensor, self._input_tensors[0].ffhandle, self._output_tensor.ffhandle)
    
  def fit(self, input_tensors, label_tensor, epochs=1):
    assert isinstance(input_tensors, list) == False, "do not support multiple inputs"
    assert self._output_tensor.ffhandle != 0, "tensor is not init"
    input_tensors = [input_tensors]
    self._verify_tensors(input_tensors, label_tensor)
        
    self._create_data_loaders(input_tensors, label_tensor)
    
    #self._init_inout() 
    self._set_optimizer()     
    self.ffmodel.init_layers()
    
    self._train(epochs)
    
  def __add_model(self, model):
    for layer_id in model.layers:
      layer = model.layers[layer_id]
      layer.reset_connection()
      self.__add_layer(layer)