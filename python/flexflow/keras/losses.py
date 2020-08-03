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
    
  