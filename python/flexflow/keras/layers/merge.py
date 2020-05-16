import flexflow.core as ff

from .base_layer import Layer
from flexflow.keras.models.input_layer import Tensor, Input

class Concatenate(Layer):
  def __init__(self, axis, name="concatenate"):
    super(Concatenate, self).__init__(name, "Concatenate") 
    
    self.axis = axis
    self.input_shape = (0, 0, 0, 0)
    self.output_shape = (0, 0, 0, 0)
    
  def calculate_inout_shape(self, input_tensors):
    if (input_tensors[0].num_dims == 2):
      output_shape = [input_tensors[0].batch_shape[0], 0]
      for input_tensor in input_tensors:
        output_shape[self.axis] += input_tensor.batch_shape[self.axis]
      self.output_shape = (output_shape[0], output_shape[1])
    elif (input_tensors[0].num_dims == 4):
      output_shape = [input_tensors[0].batch_shape[0], 0, input_tensors[0].batch_shape[2], input_tensors[0].batch_shape[3]]
      for input_tensor in input_tensors:
        output_shape[self.axis] += input_tensor.batch_shape[self.axis]
      self.output_shape = (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    else:
      assert 0, "un-supported dims"
    print("concat output ", self.output_shape)
  
  def verify_meta_data(self):
   pass
  
  def verify_inout_shape(self, input_tensor, output_tensor):
    pass
    
  def get_summary(self):
    summary = "%s%s%s\n"%(self._get_summary_name(), self.output_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensors):
    self.calculate_inout_shape(input_tensors)
    output_tensor = Tensor(batch_shape=self.output_shape, dtype=input_tensors[0].dtype, meta_only=True)
    self.output_tensor = output_tensor
    
    output_tensor.set_from_layer(self)
    
    for tensor in input_tensors:
      self.input_tensors.append(tensor)
      assert tensor.from_layer != 0, "check input tensor"
      self.prev_layers.append(tensor.from_layer)
      tensor.from_layer.next_layers.append(self)
    return output_tensor