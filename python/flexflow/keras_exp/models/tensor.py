# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

class Tensor(object):
  def __init__(self, ffconfig=None,
               key=0,
               shape=None,
               batch_shape=None,
               dtype=None):

    self._ffhandle = None
    if dtype == None or dtype == "float32" or dtype == ff.DataType.DT_FLOAT:
      self.dtype = ff.DataType.DT_FLOAT
    elif dtype == "float64" or dtype == ff.DataType.DT_DOUBLE:
      self.dtype = ff.DataType.DT_DOUBLE
    elif dtype == "int32" or dtype == ff.DataType.DT_INT32:
      self.dtype = ff.DataType.DT_INT32
    elif dtype == "int64" or dtype == ff.DataType.DT_INT64:
      self.dtype = ff.DataType.DT_INT64
    else:
      assert 0, "not supported"
      
    if batch_shape != None:
      self.batch_shape = batch_shape
    else:
      self.batch_shape = (ffconfig.batch_size,) + tuple(shape[1:])
    #print(self.batch_shape)
    self.num_dims = len(self.batch_shape)
    self.key = key

  @property
  def ffhandle(self):
    return self._ffhandle
    
  @ffhandle.setter
  def ffhandle(self, handle):
    assert isinstance(handle, ff.Tensor) == True, "[Tensor]: ffhandle is not the correct type"
    assert self._ffhandle == None, "[Tensor]: check handle, already set"
    self._ffhandle = handle
    self.__verify_ffhandle_shape()
    self.__verify_ffhandle_dtype()

  @property
  def dtype_str(self):
    if self.dtype == ff.DataType.DT_FLOAT:
      return "float32"
    elif self.dtype == ff.DataType.DT_DOUBLE:
      return "float64"
    elif self.dtype == ff.DataType.DT_INT32:
      return "int32"
    elif self.dtype == ff.DataType.DT_INT64:
      return "int64"

  def create_ff_tensor(self, ffmodel):
    assert self.batch_shape[0] != 0, "[Tensor]: batch size is not set"
    if (self.num_dims == 2 or self.num_dims == 4):
      #print(self.batch_shape, type(self.batch_shape))
      self._ffhandle = ffmodel.create_tensor(self.batch_shape, self.dtype);
    else:
      assert 0, "un-supported dims"
    self.__verify_ffhandle_shape()
    self.__verify_ffhandle_dtype()

  def set_batch_size(self, size):
    lst = list(self.batch_shape)
    lst[0] = size
    self.batch_shape = tuple(lst)

  def __verify_ffhandle_shape(self):
    assert self.num_dims == self._ffhandle.num_dims, "[Tensor]: check tensor shape"
    for i in range(0, self.num_dims):
      assert self.batch_shape[i] == self._ffhandle.dims[i], "[Tensor]: please check shape dim %d (%d == %d)" %(i, self.batch_shape[i], self._ffhandle.dims[i])

  def __verify_ffhandle_dtype(self):
    assert self.dtype == self._ffhandle.data_type