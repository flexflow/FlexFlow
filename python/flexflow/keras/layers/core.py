import flexflow.core as ff

class Dense(object):
  def __init__(self, output_shape, input_shape, activation=None):
    self.out_channels = output_shape
    self.in_channels = input_shape[0]
    if (activation == "relu"):
      self.activation = ff.ActiMode.AC_MODE_RELU
    else:
      self.activation = ff.ActiMode.AC_MODE_NONE
    self.handle = 0
    self.name = "dense"
    
class Flatten(object):
  def __init__(self):
    self.handle = 0
    self.name = "flat"
    
class Activation(object):
  def __init__(self, type):
    if (type == "softmax"):
      self.type = "softmax"