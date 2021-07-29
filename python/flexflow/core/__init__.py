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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import atexit
import os
import sys

from .flexflow_env import *

# check which python binding to use
if flexflow_python_binding() == 'pybind11':
  print("Using pybind11 flexflow bindings.")
  from .flexflow_pybind11 import *
else:
  print("Using cffi flexflow bindings.")
  from .flexflow_cffi import *
  from .flexflow_type import *

# check if use native python interpreter
if flexflow_python_interpreter() == 'native':  
  from .flexflow_pybind11_internal import begin_flexflow_task, finish_flexflow_task
  print("Using native python")
  begin_flexflow_task(sys.argv)
  atexit.register(finish_flexflow_task)
else:
  print("Using flexflow python")

#from flexflow.core.flexflow_logger import *
if 'FF_BUILD_DOCS' not in os.environ:
  build_docs = 0
else:
  build_docs = int(os.environ['FF_BUILD_DOCS'])
if build_docs == 1:
  pass
else:
  from .flexflow_top import flexflow_top_level_task, get_legion_runtime, get_legion_context
