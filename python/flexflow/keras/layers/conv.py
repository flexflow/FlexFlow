import flexflow.core as ff
import math

from .base_layer import Layer

import builtins

class Conv2D(Layer):
  def __init__(self, filters, input_shape=(0,), kernel_size=0, strides=0, padding=0, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
    super(Conv2D, self).__init__("conv2d") 
    
    self.input_shape = (0, 0, 0, 0)
    self.output_shape = (0, 0, 0, 0)
    self.out_channels = filters
    assert len(kernel_size)==2, "wrong dim of kernel_size"
    self.kernel_size = kernel_size
    assert len(strides)==2, "wrong dim of stride"
    self.stride = strides
    self.padding = padding
    if (activation == None):
      self.activation = ff.ActiMode.AC_MODE_NONE
    elif(activation =="relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    else:
      self.activation = ff.ActiMode.AC_MODE_NONE
    if (len(input_shape) == 4):
      self.in_channels = input_shape[1]
      self.input_shape = input_shape
      self.calculate_inout_shape(input_shape[1], input_shape[2], input_shape[3], input_shape[0])
    elif (len(input_shape) == 3):
      self.in_channels = input_shape[0]
      self.input_shape = (0, input_shape[0], input_shape[1], input_shape[2])
      self.calculate_inout_shape(input_shape[0], input_shape[1], input_shape[2])
    else:
      self.in_channels = 0
    self.use_bias = use_bias
    
  def calculate_inout_shape(self, input_d, input_w, input_h, input_b=0):
    assert input_w != 0, "wrong input_w"
    assert input_h != 0, "wrong input_h"
    assert input_d != 0, "wrong input_d"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == input_d, "wrong input_w"
      assert self.input_shape[1] == input_d, "wrong input_w"
      assert self.input_shape[2] == input_w, "wrong input_h"
      assert self.input_shape[3] == input_h, "wrong input_d"
    self.input_shape = (input_b, input_d, input_w, input_h)
    self.in_channels = input_d
    output_w = 1 + math.floor((input_w + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
    output_h = 1 + math.floor((input_h + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
    output_d = self.out_channels
    self.output_shape = (input_b, output_d, output_w, output_h)
    print("conv2d input ", self.input_shape)
    print("conv2d output ", self.output_shape)
    
  def verify_inout_shape(self, input_tensor, output_tensor):
    in_dims = input_tensor.dims
    assert in_dims[1] == self.input_shape[1]
    assert in_dims[2] == self.input_shape[2]
    assert in_dims[3] == self.input_shape[3]
    out_dims = output_tensor.dims
    assert out_dims[1] == self.output_shape[1]
    assert out_dims[2] == self.output_shape[2]
    assert out_dims[3] == self.output_shape[3]
    
  def __call__(self, input_tensor):
    output_tensor = builtins.internal_ffmodel.conv2d(self.name, input_tensor, self.out_channels, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.padding[0], self.padding[1], self.activation, self.use_bias)
    return output_tensor
    
  def get_weights(self, ffmodel):
    return self._get_weights(ffmodel)
    
  def set_weights(self, ffmodel, kernel, bias):
    self._set_weights(ffmodel, kernel, bias)