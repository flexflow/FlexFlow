import flexflow.core as ff

from .base_layer import Layer
from flexflow.keras.models.input_layer import Tensor

import builtins

class Dense(Layer):
  def __init__(self, output_shape, input_shape=(0,), activation=None):
    super(Dense, self).__init__("dense") 
    
    self.out_channels = output_shape
    self.in_channels = 0
    self.input_shape = (0, 0)
    self.output_shape = (0, 0)
    if (len(input_shape) == 2):
      self.in_channels = input_shape[1]
      self.calculate_inout_shape(input_shape[1], input_shape[0])
    elif (len(input_shape) == 1):
      if (input_shape[0] != 0):
        self.in_channels = input_shape[0]
        self.calculate_inout_shape(input_shape[0])
      else:
        self.in_channels = 0
    if (activation == None):
      self.activation = ff.ActiMode.AC_MODE_NONE
    elif(activation =="relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    else:
      assert 0, "activation is not supported"
  
  def calculate_inout_shape(self, in_dim, input_b=0):
    assert in_dim != 0, "wrong in_dim"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == in_dim, "wrong input_w"
    self.output_shape = (input_b, self.out_channels)
    self.input_shape = (input_b, in_dim)
    self.in_channels = in_dim
    print("dense input ", self.input_shape)
    print("dense output ", self.output_shape)
    
  def verify_meta_data(self):
    assert self.input_shape != (0, 0), "input shape is wrong"
    assert self.output_shape != (0, 0), "output shape is wrong"
    assert self.in_channels != 0, " in channels is wrong"
    assert self.out_channels != 0, " out channels is wrong"
    
  def verify_inout_shape(self, input_tensor_handle, output_tensor_handle):
    in_dims = input_tensor_handle.dims
    assert in_dims[1] == self.input_shape[1], "%d, %d" % (in_dims[0], self.input_shape[1])
    print(in_dims[1], self.input_shape[1])
    out_dims = output_tensor_handle.dims
    assert out_dims[1] == self.output_shape[1]
    
  def get_summary(self):
    summary = "%s (Dense)\t\t%s\t\t%s\n"%(self.name, self.output_shape, self.input_shape)
    return summary
    
  def __call__(self, input_tensor):
    in_dims = input_tensor.batch_shape
    self.calculate_inout_shape(in_dims[1], in_dims[0])
    output_tensor = Tensor(batch_shape=self.output_shape, dtype=input_tensor.dtype, meta_only=True)
    input_tensor.input_layers.append(self)
    output_tensor.output_layers.append(self)
    if (len(input_tensor.output_layers) != 0):
      assert len(input_tensor.output_layers) == 1, "check input tensor"
      self.prev_layers.append(input_tensor.output_layers[0])
      input_tensor.output_layers[0].next_layers.append(self)
    return output_tensor
    
  def get_weights(self, ffmodel):
    return self._get_weights(ffmodel)
    
  def set_weights(self, ffmodel, kernel, bias):
    self._set_weights(ffmodel, kernel, bias)
    
class Flatten(Layer):
  def __init__(self):
    super(Flatten, self).__init__("flat") 
    self.input_shape = 0
    self.output_shape = (0, 0)
    
  def calculate_inout_shape(self, input_shape):
    self.input_shape = input_shape
    flat_size = 1
    for i in range(1, len(input_shape)):
      flat_size *= input_shape[i]
    self.output_shape = (input_shape[0], flat_size)
    print("flat input ", self.input_shape)
    print("flat output ", self.output_shape)
    
  def verify_meta_data(self):
    assert self.input_shape != 0, "input shape is wrong"
    assert self.output_shape != (0, 0), "output shape is wrong"
    
  def verify_inout_shape(self, input_tensor_handle, output_tensor_handle):
    out_dims = output_tensor_handle.dims
    assert out_dims[1] == self.output_shape[1]
    
  def get_summary(self):
    summary = "%s (Flatten)\t\t%s\t\t%s\n"%(self.name, self.output_shape, self.input_shape)
    return summary
    
  def __call__(self, input_tensor):
    in_dims = input_tensor.batch_shape
    self.calculate_inout_shape(in_dims)
    output_tensor = Tensor(batch_shape=self.output_shape, dtype=input_tensor.dtype, meta_only=True)
    input_tensor.input_layers.append(self)
    output_tensor.output_layers.append(self)
    assert len(input_tensor.output_layers) == 1, "check input tensor"
    self.prev_layers.append(input_tensor.output_layers[0])
    input_tensor.output_layers[0].next_layers.append(self)
    return output_tensor
    
class Activation(Layer):
  def __init__(self, type):
    super(Activation, self).__init__("activation") 
    
    if (type == "softmax"):
      self.type = "softmax"
      
  def verify_meta_data(self):
    assert self.type == "softmax", "type is wrong"
      
  def verify_inout_shape(self, input_tensor, output_tensor):
    v = 1
    
  def get_summary(self):
    summary = "%s (Softmax)\n"%(self.name)
    return summary
    
  def __call__(self, input_tensor):
    output_tensor = Tensor(batch_shape=input_tensor.batch_shape, dtype=input_tensor.dtype, meta_only=True)
    input_tensor.input_layers.append(self)
    output_tensor.output_layers.append(self)
    assert len(input_tensor.output_layers) == 1, "check input tensor"
    self.prev_layers.append(input_tensor.output_layers[0])
    input_tensor.output_layers[0].next_layers.append(self)
    return output_tensor