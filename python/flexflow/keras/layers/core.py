import flexflow.core as ff

class Dense(object):
  def __init__(self, output_shape, input_shape=(0,), activation=None):
    self.out_channels = output_shape
    self.in_channels = 0
    self.input_shape = (0, 0)
    self.output_shape = (0, 0)
    if (len(input_shape) == 2):
      self.in_channels = input_shape[0]
      self.calculate_inout_shape(input_shape[0], input_shape[1])
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
    self.handle = 0
    self.name = "dense"
  
  def calculate_inout_shape(self, in_dim, input_b=0):
    assert in_dim != 0, "wrong in_dim"
    if (self.in_channels != 0): # check if user input is correct
      assert self.in_channels == in_dim, "wrong input_w"
    self.output_shape = (self.out_channels, input_b)
    self.input_shape = (in_dim, input_b)
    self.in_channels = in_dim
    print("dense input ", self.input_shape)
    print("dense output ", self.output_shape)
    
class Flatten(object):
  def __init__(self):
    self.input_shape = 0
    self.output_shape = (0, 0)
    self.handle = 0
    self.name = "flat"
    
  def calculate_inout_shape(self, input_shape):
    self.input_shape = input_shape
    flat_size = 1
    for i in range(len(input_shape)-1):
      flat_size *= input_shape[i]
    self.output_shape = (flat_size, input_shape[len(input_shape)-1])
    print("flat input ", self.input_shape)
    print("flat output ", self.output_shape)
    
class Activation(object):
  def __init__(self, type):
    if (type == "softmax"):
      self.type = "softmax"