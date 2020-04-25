import flexflow.core as ff
import math

import builtins

class MaxPooling2D(object):
  def __init__(self, pool_size, strides, padding="valid"):
    self.input_shape = (0, 0, 0, 0)
    self.output_shape = (0, 0, 0, 0)
    self.out_channels = 0
    self.in_channels = 0
    assert len(pool_size)==2, "wrong dim of pool_size"
    self.kernel_size = pool_size
    assert len(strides)==2, "wrong dim of strides"
    self.stride = strides
    if (padding == "valid"):
      self.padding = (0, 0)
    else:
      self.padding = (0, 0)
    self.handle = 0
    self.name = "pool2d"
    
  def calculate_inout_shape(self, input_w, input_h, input_d, input_b=0):
    assert input_w != 0, "wrong input_w"
    assert input_h != 0, "wrong input_h"
    assert input_d != 0, "wrong input_d"
    self.input_shape = (input_b, input_w, input_h, input_d)
    self.in_channels = input_d
    self.out_channels = input_d
    output_w = 1 + math.floor((input_w + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
    output_h = 1 + math.floor((input_h + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
    output_d = self.out_channels
    self.output_shape = (input_b, output_w, output_h, output_d)
    print("pool2d input ", self.input_shape)
    print("pool2d output ", self.output_shape)
    
  def verify_inout_shape(self, input_tensor, output_tensor):
    in_dims = input_tensor.dims
    assert in_dims[0] == self.input_shape[1]
    assert in_dims[1] == self.input_shape[2]
    assert in_dims[2] == self.input_shape[3]
    out_dims = output_tensor.dims
    assert out_dims[0] == self.output_shape[1]
    assert out_dims[1] == self.output_shape[2]
    assert out_dims[2] == self.output_shape[3]
    
  def __call__(self, input_tensor):
    output_tensor = builtins.internal_ffmodel.pool2d(self.name, input_tensor, self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.padding[0], self.padding[1])
    return output_tensor