import flexflow.core as ff
import math

class Conv2D(object):
  def __init__(self, filters, input_shape=(0,), kernel_size=0, strides=0, padding=0, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
    self.layer_id = -1
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
    else:
      self.activation = ff.ActiMode.AC_MODE_NONE
    if (len(input_shape) == 4):
      self.in_channels = input_shape[2]
      self.input_shape = input_shape
      self.calculate_inout_shape(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    elif (len(input_shape) == 3):
      self.in_channels = input_shape[2]
      self.input_shape = (input_shape[0], input_shape[1], input_shape[2], 0)
      self.calculate_inout_shape(input_shape[0], input_shape[1], input_shape[2])
    else:
      self.in_channels = 0
    self.use_bias = use_bias
    self.handle = 0
    self.name = "conv2d"
    
  def calculate_inout_shape(self, input_w, input_h, input_d, input_b=0):
    assert input_w != 0, "wrong input_w"
    assert input_h != 0, "wrong input_h"
    assert input_d != 0, "wrong input_d"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == input_d, "wrong input_w"
      assert self.input_shape[0] == input_w, "wrong input_w"
      assert self.input_shape[1] == input_h, "wrong input_h"
      assert self.input_shape[2] == input_d, "wrong input_d"
    self.input_shape = (input_w, input_h, input_d, input_b)
    self.in_channels = input_d
    output_w = 1 + math.floor((input_w + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0])
    output_h = 1 + math.floor((input_h + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1])
    output_d = self.out_channels
    self.output_shape = (output_w, output_h, output_d, input_b)
    print("conv2d input ", self.input_shape)
    print("conv2d output ", self.output_shape)
    
  def __call__(self, input):
    print("conv2d call")
    return 1