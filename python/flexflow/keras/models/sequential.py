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
    
    prev_layer = 0
    
    if (layer.layer_id == 0):
      input_shape = list(layer.input_shape)
      input_tensor = Input(batch_shape=input_shape, dtype="float32")
      self.input_tensors.append(input_tensor)
      self.output_tensor = input_tensor
      
    self.output_tensor = layer(self.output_tensor)
    
    layer.verify_meta_data()
    
  def compile(self, optimizer):
    self.input_tensors[0].set_batch_size(self.ffconfig.get_batch_size())
    self._create_input_tensor(0)
    self._create_label_tensor()
    self._create_flexflow_layers()
    self._compile(optimizer)
    
  def fit(self, input_tensors, label_tensor, epochs=1):
    assert isinstance(input_tensors, list) == False, "do not support multiple inputs"
    input_tensors = [input_tensors]
    self._verify_tensors(input_tensors, label_tensor)
        
    self._create_data_loaders(input_tensors, label_tensor)
    
    #self.__init_inout() 
    self._set_optimizer()     
    self.ffmodel.init_layers()
    
    self._train(epochs)
    
  def _create_flexflow_layers(self, verify_inout_shape=True):
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]

      if (isinstance(layer, Activation) == True):
       assert layer.layer_id == self._nb_layers-1, "softmax is not in the last layer"
       out_t = self.ffmodel.softmax("softmax", layer.input_tensors[0].ffhandle, self.label_tensor.ffhandle)
      elif (isinstance(layer, Concatenate) == True):
       t_ffhandle_list = []
       for t in layer.input_tensors:
         t_ffhandle_list.append(t.ffhandle)
       out_t = self.ffmodel.concat("concat", t_ffhandle_list, layer.axis)
      elif (isinstance(layer, Conv2D) == True):
       out_t = self.ffmodel.conv2d(layer.name, layer.input_tensors[0].ffhandle, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
      elif (isinstance(layer, MaxPooling2D) == True):
       out_t = self.ffmodel.pool2d(layer.name, layer.input_tensors[0].ffhandle, layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
      elif (isinstance(layer, Flatten) == True):
       out_t = self.ffmodel.flat(layer.name, layer.input_tensors[0].ffhandle)
      elif (isinstance(layer, Dense) == True):
       out_t = self.ffmodel.dense(layer.name, layer.input_tensors[0].ffhandle, layer.out_channels, layer.activation)
      else:
       assert 0, "unknow layer"

      layer.output_tensor.set_ffhandle(out_t)

      assert layer.ffhandle == 0, "layer handle is inited"
      layer.ffhandle = self.ffmodel.get_layer_by_id(layer.layer_id)
      assert layer.ffhandle != 0, "layer handle is wrong"
      print(layer.ffhandle)    

      if (verify_inout_shape == True):
       in_t = layer.input_tensors[0].ffhandle
       layer.verify_inout_shape(in_t, out_t)
    
  def __create_flexflow_layers_v2(self):
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      if (isinstance(layer, Conv2D) == True):
        layer.ffhandle = self.ffmodel.conv2d_v2(layer.name, layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.kernel_size[1], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1], layer.activation, layer.use_bias)
      elif (isinstance(layer, MaxPooling2D) == True):
        layer.ffhandle = self.ffmodel.pool2d_v2(layer.name, layer.kernel_size[1], layer.kernel_size[0], layer.stride[0], layer.stride[1], layer.padding[0], layer.padding[1])
      elif (isinstance(layer, Flatten) == True):
        layer.ffhandle = self.ffmodel.flat_v2(layer.name)
      elif (isinstance(layer, Dense) == True):
        layer.ffhandle = self.ffmodel.dense_v2(layer.name, layer.in_channels, layer.out_channels, layer.activation)
      elif (isinstance(layer, Activation) == True):
        print("add softmax")
      else:
        assert 0, "unknow layer"
    
  def __init_inout(self, verify_inout_shape=True):
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]

      if (isinstance(layer, Activation) == True):
        assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
        out_t = self.ffmodel.softmax("softmax", layer.input_tensors[0].ffhandle, self.label_tensor.ffhandle)
        assert layer.ffhandle == 0, "layer handle is inited"
        layer.ffhandle = self.ffmodel.get_layer_by_id(layer_id)
      else:
        out_t = layer.ffhandle.init_inout(self.ffmodel, layer.input_tensors[0].ffhandle);
      
      layer.output_tensor.set_ffhandle(out_t)
      assert layer.ffhandle != 0, "layer handle is wrong"
      print(layer.ffhandle)    
      
      if (verify_inout_shape == True):
        in_t = layer.input_tensors[0].ffhandle
        layer.verify_inout_shape(in_t, out_t)
    print("output tensor", self.output_tensor.batch_shape)
    
    