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

class Loss(object):
  def __init__(self, name=None):
    self.type = None
    self.name = name
    
class CategoricalCrossentropy(Loss):
  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction='auto',
               name='categorical_crossentropy'):
    super(CategoricalCrossentropy, self).__init__(name=name)
    self.type = ff.LossType.LOSS_CATEGORICAL_CROSSENTROPY

class SparseCategoricalCrossentropy(Loss):
  def __init__(self,
               from_logits=False,
               reduction='auto',
               name='sparse_categorical_crossentropy'):
    super(SparseCategoricalCrossentropy, self).__init__(name=name)
    self.type = ff.LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY
    
class MeanSquaredError(Loss):
  def __init__(self,
               reduction='auto',
               name='mean_squared_error'):
    super(MeanSquaredError, self).__init__(name=name)
    self.type = ff.LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE           
    
  