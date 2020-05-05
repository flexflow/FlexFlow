import flexflow.core as ff

from .base_layer import Layer

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
    if (activation == "relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    else:
      self.activation = ff.ActiMode.AC_MODE_NONE
  
  def calculate_inout_shape(self, in_dim, input_b=0):
    assert in_dim != 0, "wrong in_dim"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == in_dim, "wrong input_w"
    self.output_shape = (input_b, self.out_channels)
    self.input_shape = (input_b, in_dim)
    self.in_channels = in_dim
    print("dense input ", self.input_shape)
    print("dense output ", self.output_shape)
    
  def verify_inout_shape(self, input_tensor, output_tensor):
    in_dims = input_tensor.dims
    assert in_dims[1] == self.input_shape[1], "%d, %d" % (in_dims[0], self.input_shape[1])
    print(in_dims[1], self.input_shape[1])
    out_dims = output_tensor.dims
    assert out_dims[1] == self.output_shape[1]
    
  def __call__(self, input_tensor):
    output_tensor = builtins.internal_ffmodel.dense(self.name, input_tensor, self.out_channels, self.activation)
    in_dims = input_tensor.dims
    self.input_shape = (in_dims[0], in_dims[1])
    out_dims = output_tensor.dims
    self.output_shape = (out_dims[0], out_dims[1])
    self.in_channels = in_dims[1]
    self.outchannels = out_dims[1]
    self.verify_inout_shape(input_tensor, output_tensor)
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
    
  def verify_inout_shape(self, input_tensor, output_tensor):
    out_dims = output_tensor.dims
    assert out_dims[1] == self.output_shape[1]
    
  def __call__(self, input_tensor):
    output_tensor = builtins.internal_ffmodel.flat(self.name, input_tensor)
    return output_tensor
    
class Activation(Layer):
  def __init__(self, type):
    super(Activation, self).__init__("activation") 
    
    if (type == "softmax"):
      self.type = "softmax"
      
  def verify_inout_shape(self, input_tensor, output_tensor):
    v = 1