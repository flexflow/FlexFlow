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

class Initializer(object):
  def __init__(self):
    self._ffhandle = None
    
  @property
  def ffhandle(self):
    return self._ffhandle
    
class DefaultInitializer(Initializer):
  def __init__(self):
    super(DefaultInitializer, self).__init__() 
    self._ffhandle = None
    
class Zeros(Initializer):
  def __init__(self):
    super(Zeros, self).__init__() 
    self._ffhandle = ff.ZeroInitializer()
    
class GlorotUniform(Initializer):
  def __init__(self, seed):
    self.seed = seed
    super(GlorotUniform, self).__init__() 
    self._ffhandle = ff.GlorotUniformInitializer(self.seed)
    
class RandomUniform(Initializer):
  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    super(RandomUniform, self).__init__() 
    self._ffhandle = ff.UniformInitializer(self.seed, self.minval, self.maxval)
    
class RandomNormal(Initializer):
  def __init__(self, mean=0., stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self._ffhandle = ff.UniformInitializer(self.seed, self.mean, self.stddev)