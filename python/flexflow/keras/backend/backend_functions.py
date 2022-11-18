# Copyright 2020 Stanford University, Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import flexflow.core as ff
from flexflow.core.flexflow_logger import fflogger

from ..layers.base_layer import Layer
from ..layers.input_layer import Input
from flexflow.keras.models.tensor import Tensor


class BatchMatmul(Layer):
  def __init__(self, **kwargs):
    super(BatchMatmul, self).__init__("batch_matmul", "Batch_Matmul", **kwargs) 

  def verify_meta_data(self):
   pass

  def _calculate_inout_shape(self, input_tensors):
    if len(input_tensors) != 2:
        raise ValueError(f'Expected 2 tensors, Got {len(input_tensors)}')

    self.input_shape = input_tensors[0].batch_shape

    self.output_shape = (
        input_tensors[0].batch_shape[0], input_tensors[0].batch_shape[1],
        input_tensors[1].batch_shape[2])
    fflogger.debug("add output %s" %( str(self.output_shape)))

  def get_summary(self):
    summary = "%s%s%s\n"%(self._get_summary_name(), self.output_shape, self._get_summary_connected_to())
    return summary

  def __call__(self, input_tensors):
    return self._connect_layer_n_input_1_output(input_tensors)

  def _verify_inout_tensor_shape(self, input_tensors, output_tensor):
    if input_tensors[0].num_dims != 3 or input_tensors[1].num_dims != 3:
        raise NotImplementedError(
            'BatchMatmul is only implemented for 3 dimensional tensors. '
            f'Got {input_tensors[0].num_dims} and {input_tensors[1].num_dims}')
    if input_tensors[0].batch_shape[2] != input_tensors[1].batch_shape[1]:
        raise ValueError(
            f'Input tensors of shapes {input_tensors[0].batch_shape} and '
            f'{input_tensors[1].batch_shape} cannot be multiplied. '
            f'{input_tensors[0].batch_shape[2]} != '
            f'{input_tensors[1].batch_shape[1]}')

    for input_tensor in input_tensors:
      assert input_tensor.num_dims == len(self.input_shape), "[BatchMatmul]: check input tensor dims"
    #   for i in range (1, input_tensor.num_dims):
    #     assert input_tensor.batch_shape[i] == self.input_shape[i]
    assert output_tensor.num_dims == len(self.output_shape), "[BatchMatmul]: check output tensor dims"
    for i in range (1, output_tensor.num_dims):
      assert output_tensor.batch_shape[i] == self.output_shape[i]

  def _reset_layer(self):
    pass


def batch_dot(x, y):
    return BatchMatmul()([x, y])
