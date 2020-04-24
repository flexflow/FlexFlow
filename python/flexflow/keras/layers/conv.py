import flexflow.core as ff

class Conv2D(object):
  def __init__(self, filters, input_shape, kernel_size, strides, padding):
    self.out_channels = filters
    if (len(input_shape) == 3):
      self.in_channels = input_shape[2]
    else:
      self.in_channels = input_shape[0]
    self.kernel_size = kernel_size[0]
    self.stride = strides[0]
    self.padding = padding[0]
    self.handle = 0
    self.name = "conv2d"