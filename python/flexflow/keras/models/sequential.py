import flexflow.core as ff

from .base_model import BaseModel
from .input_layer import Tensor
from flexflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

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

    if (isinstance(layer, Conv2D) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        assert len(prev_layer.output_shape) == 4, "check prev layer"
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, MaxPooling2D) == True):
      assert layer.layer_id != 0, "maxpool2d can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      assert len(prev_layer.output_shape) == 4, "check prev layer"
      layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[2], prev_layer.output_shape[3], prev_layer.output_shape[0])
    elif (isinstance(layer, Flatten) == True):
      assert layer.layer_id != 0, "flatten can not be the 1st layer"
      prev_layer = self._layers[layer.layer_id-1]
      layer.calculate_inout_shape(prev_layer.output_shape)
    elif (isinstance(layer, Dense) == True):
      if (layer.layer_id > 0):
        prev_layer = self._layers[layer.layer_id-1]
        assert len(prev_layer.output_shape) == 2, "check prev layer"
        layer.calculate_inout_shape(prev_layer.output_shape[1], prev_layer.output_shape[0])
    else:
      prev_layer = self._layers[layer.layer_id-1]
    
    layer.verify_meta_data()
    
    if (prev_layer != 0):
      layer.add_prev_layer(prev_layer)
      prev_layer.add_next_layer(layer)
    
  def create_input_tensor(self, input_shape):
    if (len(input_shape) == 2):
      self.input_tensors.append(Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), input_shape[1]], name="", dtype="float32"))
      
    elif (len(input_shape) == 4):
      self.input_tensors.append(Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), input_shape[1], input_shape[2], input_shape[3]], name="", dtype="float32"))
    
  def create_label_tensor(self, label_shape):
    self.label_tensor = Tensor(self.ffmodel, batch_shape=[self.ffconfig.get_batch_size(), 1], name="", dtype="int32")
    
  def compile(self, optimizer):
    self._compile(optimizer)
    self.__create_flexflow_layers()
    
  def fit(self, input_tensors, label_tensor, epochs=1):
    assert isinstance(input_tensors, list) == False, "do not support multiple inputs"
    input_tensors = [input_tensors]
    for input_tensor in input_tensors:
      self.create_input_tensor(input_tensor.shape)
    self.create_label_tensor(label_tensor.shape)
    self._verify_tensors(input_tensors, label_tensor)
        
    self._create_data_loaders(input_tensors, label_tensor)
    
    self.__init_inout() 
    self._set_optimizer()     
    self.ffmodel.init_layers()
    
    self._train(epochs)
    
  def __create_flexflow_layers(self):
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
    int_t = 0
    out_t = 0
    for layer_id in self._layers:
      layer = self._layers[layer_id]
      if (layer_id == 0):
        in_t = self.input_tensors[0].ffhandle
        out_t = layer.ffhandle.init_inout(self.ffmodel, in_t);
      else:
        in_t = out_t
        if (isinstance(layer, Activation) == True):
          assert layer_id == self._nb_layers-1, "softmax is not in the last layer"
          out_t = self.ffmodel.softmax("softmax", in_t, self.label_tensor.ffhandle)
          assert layer.ffhandle == 0, "layer handle is inited"
          layer.ffhandle = self.ffmodel.get_layer_by_id(layer_id)
        else:
          out_t = layer.ffhandle.init_inout(self.ffmodel, in_t);
      
      assert layer.ffhandle != 0, "layer handle is wrong"
      print(layer.ffhandle)
      assert layer.output_tensor == 0, "wrong"
      assert len(layer.input_tensors) == 0, "wrong"    
      
      if (verify_inout_shape == True):
        layer.verify_inout_shape(in_t, out_t)
    self.output_tensor = Tensor(dtype=self.input_tensors[0].dtype, ffhandle=out_t)
    print("output tensor", self.output_tensor.batch_shape)
    
    