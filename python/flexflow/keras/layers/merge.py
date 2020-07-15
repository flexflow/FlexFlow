import flexflow.core as ff

from .base_layer import Layer
from flexflow.keras.models.input_layer import Tensor, Input

class _Merge(Layer):
  def __init__(self, name, layer_type):
    super(_Merge, self).__init__(name, layer_type) 
  
  def verify_meta_data(self):
   pass
    
  def get_summary(self):
    summary = "%s%s%s\n"%(self._get_summary_name(), self.output_shape, self._get_summary_connected_to())
    return summary
    
  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)
    
  def _verify_inout_tensor_shape(self, input_tensors, output_tensor):
    for input_tensor in input_tensors:
      assert input_tensor.num_dims == len(self.input_shape), "[Concatenate]: check input tensor dims"
      for i in range (1, input_tensor.num_dims):
        print(input_tensor.batch_shape[i], self.input_shape[i], i)
        assert input_tensor.batch_shape[i] == self.input_shape[i]
    assert output_tensor.num_dims == len(self.output_shape), "[Concatenate]: check output tensor dims"
    for i in range (1, output_tensor.num_dims):
      assert output_tensor.batch_shape[i] == self.output_shape[i]
      
  def _reset_layer(self):
    pass
    
def concatenate(input_tensors, _axis=1):
  return Concatenate(axis=_axis)(input_tensors)
    
class Concatenate(_Merge):
  __slots__ = ['axis']
  def __init__(self, axis, name="concatenate"):
    super(Concatenate, self).__init__(name, "Concatenate") 
    
    self.axis = axis
    
  def _calculate_inout_shape(self, input_tensors):
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
    self.input_shape = input_tensors[0].batch_shape

def add(input_tensors):
  return Add()(input_tensors)
    
class Add(_Merge):
  def __init__(self, name="add"):
    super(Add, self).__init__(name, "Add") 
    
  def _calculate_inout_shape(self, input_tensors):    
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    print("add output ", self.output_shape)

def subtract(input_tensors):
  return Subtract()(input_tensors)
    
class Subtract(_Merge):
  def __init__(self, name="subtract"):
    super(Subtract, self).__init__(name, "Subtract") 
    
  def _calculate_inout_shape(self, input_tensors): 
    assert len(input_tensors) == 2, "check input_tensors"   
    self.input_shape = input_tensors[0].batch_shape
    self.output_shape = input_tensors[0].batch_shape
    print("add output ", self.output_shape)