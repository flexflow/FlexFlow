# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import atexit
import os
import sys

from flexflow.config import *

if flexflow_init_import():
  from legion_cffi import ffi, lib as legion
  from .flexflowlib import flexflow_library
  flexflow_library.initialize()

  # check which python binding to use
  if flexflow_python_binding() == 'pybind11':
    print("Using pybind11 flexflow bindings.")
    from .flexflow_pybind11 import *
  else:
    print("Using cffi flexflow bindings.")
    from .flexflow_cffi import *

else:
  pass