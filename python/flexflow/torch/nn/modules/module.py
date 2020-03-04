from collections import OrderedDict, namedtuple

from flexflow.core import *

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
  def __repr__(self):
    if not self.missing_keys and not self.unexpected_keys:
        return '<All keys matched successfully>'
    return super(_IncompatibleKeys, self).__repr__()

  __str__ = __repr__

class Module(object):
  def __init__(self):
    self.text = "Module"
    
      
  def __repr__(self):
    attrs = vars(self)
    return str(attrs)
    