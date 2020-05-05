import flexflow.core as ff

class Layer(object):
  def __init__(self, name):
    self.layer_id = -1
    self.handle = 0
    self.name = name
    
  def _get_weights(self, ffmodel):
    assert self.handle != 0, "handle is not set correctly"
    kernel_parameter = self.handle.get_weight_tensor()
    bias_parameter = self.handle.get_bias_tensor()
    kernel_array = kernel_parameter.get_weights(ffmodel)
    bias_array = bias_parameter.get_weights(ffmodel)
    return (kernel_array, bias_array)
    
  def _set_weights(self, ffmodel, kernel, bias):
    assert self.handle != 0, "handle is not set correctly"
    kernel_parameter = self.handle.get_weight_tensor()
    bias_parameter = self.handle.get_bias_tensor()
    kernel_parameter.set_weights(ffmodel, kernel)
    bias_parameter.set_weights(ffmodel, bias)
    