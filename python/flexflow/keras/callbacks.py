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