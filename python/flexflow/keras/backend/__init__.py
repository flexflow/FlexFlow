from __future__ import absolute_import
from __future__ import print_function
import sys

# Default backend: FlexFlow.
_BACKEND = 'flexflow'

# import backend
if _BACKEND == 'flexflow':
    sys.stderr.write('Using flexflow backend.\n')
    from .flexflow_backend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.
    """
    return _BACKEND