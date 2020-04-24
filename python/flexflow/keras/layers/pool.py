import flexflow.core as ff

class MaxPooling2D(object):
  def __init__(self, pool_size, strides, padding="valid"):
    self.kernel_size = pool_size[0]
    self.stride = strides[0]
    if (padding == "valid"):
      self.padding = 0
    else:
      self.padding = 0
    self.handle = 0
    self.name = "pool2d"