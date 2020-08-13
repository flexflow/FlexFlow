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
import numpy as np

from . import backend as K

class Callback(object):
  def __init__(self):
    self.validation_data = None

  def set_params(self, params):
    self.params = params

  def set_model(self, model):
    self.model = model

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def on_epoch_end(self, epoch, logs=None):
    pass

  def on_batch_begin(self, batch, logs=None):
    pass

  def on_batch_end(self, batch, logs=None):
    pass

  def on_train_begin(self, logs=None):
    pass

  def on_train_end(self, logs=None):
    pass
    
class LearningRateScheduler(Callback):
  def __init__(self, schedule):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    lr = self.schedule(epoch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                         'should be float.')
    self.model.optimizer.set_learning_rate(lr)
    print("set learning rate ", self.model.optimizer.lr)
    
class VerifyMetrics(Callback):
  def __init__(self, accuracy):
    super(VerifyMetrics, self).__init__()
    self.accuracy = accuracy.value

  def on_train_end(self, logs=None):
    perf_metrics = self.model.ffmodel.get_perf_metrics()
    accuracy = perf_metrics.get_accuracy()
    if accuracy < self.accuracy:
      assert 0, "Accuracy is wrong"
      
class EpochVerifyMetrics(Callback):
  def __init__(self, accuracy, early_stop=True):
    super(EpochVerifyMetrics, self).__init__()
    self.accuracy = accuracy.value
    self.early_stop = early_stop

  def on_epoch_end(self, logs=None):
    perf_metrics = self.model.ffmodel.get_perf_metrics()
    accuracy = perf_metrics.get_accuracy()
    if self.early_stop == False:
      return False
    if accuracy > self.accuracy:
      return True
    else:
      return False
      
    