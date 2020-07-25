import flexflow.core as ff

class SGD(object):
  def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD", **kwargs):
    self.lr = learning_rate
    self.momentum = momentum
    self.nesterov = nesterov
    self.ffhandle = None
    
  def set_learning_rate(self, learning_rate):
    self.lr = learning_rate
    self.ffhandle.set_learning_rate(learning_rate)
    
class Adam(object):
  def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
    self.lr = learning_rate
    self.beta1 = beta_1
    self.beta2 = beta_2
    self.epsilon = epsilon
    self.amsgrad = amsgrad
    self.ffhandle = None
    
  def set_learning_rate(self, learning_rate):
    self.lr = learning_rate
    self.ffhandle.set_learning_rate(learning_rate)