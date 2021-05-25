# Copyright 2021 Stanford University, Los Alamos National Laboratory
#                Facebook
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

import numpy as np
from .flexflow_logger import fflogger
# from .flexflow_type import ActiMode, AggrMode, PoolType, DataType, LossType, CompMode, MetricsType, OpType, ParameterSyncType, enum_to_int, int_to_enum
from .flexflow_pybind11_internal import ActiMode, CompMode, DataType, LossType, MetricsType, PoolType
from .flexflow_pybind11_internal import begin_flexflow_task, finish_flexflow_task
from .flexflow_pybind11_internal import Initializer, GlorotUniformInitializer, UniformInitializer, ZeroInitializer
from .flexflow_pybind11_internal import Optimizer, SGDOptimizer, AdamOptimizer
from .flexflow_pybind11_internal import Op, NetConfig, SingleDataLoader, Tensor, FFConfig, PerfMetrics
from .flexflow_pybind11_internal import FFModel as _FFModel

ff_tracing_id = 200

# -----------------------------------------------------------------------
# FFModel
# -----------------------------------------------------------------------

class FFModel(_FFModel):
  
  def __init__(self, ffconfig):
    super(FFModel, self).__init__(ffconfig)
    self._layers = dict()
    self._nb_layers = 0
    self._ffconfig = ffconfig
    global ff_tracing_id
    self._tracing_id = ff_tracing_id
    ff_tracing_id += 1
  
  def fit(self, x=None, y=None, batch_size=None, epochs=1):
    if (isinstance(x, list) == False):
      dataloaders = [x]
    else:
      dataloaders = x
    dataloaders.append(y)

    num_samples = y.num_samples
    batch_size = self._ffconfig.batch_size
    self._tracing_id += 1 # get a new tracing id
    for epoch in range(0,epochs):
      for d in dataloaders:
        d.reset()
      self.reset_metrics()
      iterations = num_samples / batch_size
      for iter in range(0, int(iterations)):
        for d in dataloaders:
          d.next_batch(self)
        self._ffconfig.begin_trace(self._tracing_id)
        self.forward()
        self.zero_gradients()
        self.backward()
        self.update()
        self._ffconfig.end_trace(self._tracing_id)
        
  def eval(self, x=None, y=None, batch_size=None):
    if (isinstance(x, list) == False):
      dataloaders = [x]
    else:
      dataloaders = x
    dataloaders.append(y)

    num_samples = y.num_samples
    batch_size = self._ffconfig.batch_size
    for d in dataloaders:
      d.reset()
    self.reset_metrics()
    iterations = num_samples / batch_size
    self._tracing_id += 1 # get a new tracing id
    for iter in range(0, int(iterations)):
      for d in dataloaders:
        d.next_batch(self)
      self._ffconfig.begin_trace(self._tracing_id)
      self.forward()
      self.compute_metrics()
      self._ffconfig.end_trace(self._tracing_id)
