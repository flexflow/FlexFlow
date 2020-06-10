import flexflow.core as ff

from .base_model import BaseModel
from .input_layer import Tensor, Input
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Concatenate

class Sequential(BaseModel):
  def __init__(self, layer_list=[]):
    super(Sequential, self).__init__() 
    
    if len(layer_list) > 0:
      for layer in layer_list:
        self.add(layer)
    
  def add(self, layer):
    self._layers[self._nb_layers] = layer
    assert layer.layer_id == -1, "layer id is inited"
    assert layer.ffhandle == 0, "layer handle is inited"
    layer.layer_id = self._nb_layers
    self._nb_layers += 1
    
    if (layer.layer_id == 0):
      input_shape = list(layer.input_shape)
      input_tensor = Input(batch_shape=input_shape, dtype="float32")
      self.input_tensors.append(input_tensor)
      self.output_tensor = input_tensor
      
    self.output_tensor = layer(self.output_tensor)
    
    layer.verify_meta_data()
    
  def compile(self, optimizer):
    self._create_input_and_label_tensors()
    self._create_flexflow_layers()
    #self._init_inout() 
    
    self._verify_output_tensors()
    self._verify_input_tensors()
    self._compile(optimizer)
    print(self.input_tensors[0], self.output_tensor, self.input_tensors[0].ffhandle, self.output_tensor.ffhandle)
    
  def fit(self, input_tensors, label_tensor, epochs=1):
    assert isinstance(input_tensors, list) == False, "do not support multiple inputs"
    input_tensors = [input_tensors]
    self._verify_tensors(input_tensors, label_tensor)
        
    self._create_data_loaders(input_tensors, label_tensor)
    
    #self._init_inout() 
    self._set_optimizer()     
    self.ffmodel.init_layers()
    
    self._train(epochs)
    
    