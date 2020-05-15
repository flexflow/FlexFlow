import flexflow.core as ff

class Layer(object):
  def __init__(self, name):
    self.layer_id = -1
    self.ffhandle = 0
    self.name = name
    self.prev_layers = []
    self.next_layers = []
    self.input_tensors = []
    self.output_tensor = 0
    self.nb_visited_prev_layers = 0
    
  def _get_weights(self, ffmodel):
    assert self.ffhandle != 0, "handle is not set correctly"
    kernel_parameter = self.ffhandle.get_weight_tensor()
    bias_parameter = self.ffhandle.get_bias_tensor()
    kernel_array = kernel_parameter.get_weights(ffmodel)
    bias_array = bias_parameter.get_weights(ffmodel)
    return (kernel_array, bias_array)
    
  def _set_weights(self, ffmodel, kernel, bias):
    assert self.ffhandle != 0, "handle is not set correctly"
    kernel_parameter = self.ffhandle.get_weight_tensor()
    bias_parameter = self.ffhandle.get_bias_tensor()
    kernel_parameter.set_weights(ffmodel, kernel)
    bias_parameter.set_weights(ffmodel, bias)
    
  def add_prev_layer(self, layer):
    self.prev_layers.append(layer)
    
  def add_next_layer(self, layer):
    self.next_layers.append(layer)
    