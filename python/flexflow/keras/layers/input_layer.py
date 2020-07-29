import flexflow.core as ff
import math

from .base_layer import Layer
from flexflow.keras.models.tensor import Tensor

class InputLayer(Layer):
  def __init__(self, shape=None, batch_size=None, 
               dtype=None, sparse=False,
               tensor=None, ragged=False,
               **kwargs):
    super(InputLayer, self).__init__("input", "InputLayer", **kwargs) 
    default_name = "input"
    if "name" in kwargs:
      default_name = kwargs["name"]
    output_tensor = Tensor(ffmodel=None, shape=shape, dtype=dtype, meta_only=True, **kwargs) 
    output_tensor.set_from_layer(self)
    self.output_tensors.append(output_tensor)
    self.output_shape = output_tensor.batch_shape
    
  def get_summary(self):
    summary = "%s%s\t\t%s\t%s\n"%(self._get_summary_name(), self.output_shape, self.input_shape, self._get_summary_connected_to())
    return summary
    
def Input(shape=None, batch_size=None, 
          dtype=None, sparse=False,
          tensor=None, ragged=False,
          **kwargs):
  input_layer = InputLayer(shape, batch_size, dtype, sparse, tensor, ragged, **kwargs)
  output_tensor = input_layer.output_tensors[0]
  return output_tensor