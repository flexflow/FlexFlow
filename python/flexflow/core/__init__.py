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
  # check which python binding to use
  if flexflow_python_binding() == 'pybind11':
    print("Using pybind11 flexflow bindings.")
    from .flexflow_pybind11 import *
  else:
    print("Using cffi flexflow bindings.")
    from .flexflow_cffi import *

  # check if use native python interpreter
  if flexflow_python_interpreter() == 'native': 
    print("Using native python")
    if flexflow_python_binding() == 'pybind11':
      from .flexflow_pybind11_internal import begin_flexflow_task, finish_flexflow_task
      print("Using native python")
      begin_flexflow_task(sys.argv)
      atexit.register(finish_flexflow_task)
    else:
      from .flexflow_cffi_header import ffc, ffi
      argv = []
      for arg in sys.argv:
        argv.append(ffi.new("char[]", arg.encode('ascii')))
      ffc.begin_flexflow_task(len(sys.argv), argv)
      atexit.register(ffc.finish_flexflow_task)
  else:
    print("Using flexflow python")
    
  from .flexflow_top import flexflow_top_level_task, get_legion_runtime, get_legion_context

else:
  pass