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

class Optimizer(object):
  def __init__(self):
    self._ffhandle = None
    
  @property
  def ffhandle(self):
    return self._ffhandle

class SGD(Optimizer):
  def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD", **kwargs):
    self.lr = learning_rate
    self.momentum = momentum
    self.nesterov = nesterov
    super(SGD, self).__init__() 
    
  def create_ffhandle(self, ffmodel):
    self._ffhandle = ff.SGDOptimizer(ffmodel, self.lr, self.momentum, self.nesterov)
    
  def set_learning_rate(self, learning_rate):
    self.lr = learning_rate
    self._ffhandle.set_learning_rate(learning_rate)
    
class Adam(Optimizer):
  def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
    self.lr = learning_rate
    self.beta1 = beta_1
    self.beta2 = beta_2
    self.epsilon = epsilon
    self.amsgrad = amsgrad
    super(Adam, self).__init__() 
    
  def create_ffhandle(self, ffmodel):
    self._ffhandle = ff.AdamOptimizer(ffmodel, self.lr, self.beta1, self.beta2, epsilon=self.epsilon)
    
  def set_learning_rate(self, learning_rate):
    self.lr = learning_rate
    self._ffhandle.set_learning_rate(learning_rate)