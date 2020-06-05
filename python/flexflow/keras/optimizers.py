import flexflow.core as ff

class SGD(object):
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.ffhandle = 0
    
class Adam(object):
  def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False):
    self.learning_rate = learning_rate
    self.beta1 = beta_1
    self.beta2 = beta_2
    self.amsgrad = amsgrad
    self.ffhandle = 0