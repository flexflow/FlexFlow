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
from flexflow.jupyter import *


if flexflow_init_import():
  os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
  from legion_cffi import ffi, is_legion_python
  from .flexflowlib import flexflow_library
  
  # Default python mode
  if is_legion_python == False:
    print("Using Default Python")
    _FF_BUILD_DOCS = bool(os.environ.get('READTHEDOCS') or os.environ.get("FF_BUILD_DOCS"))
    _CPU_ONLY = bool(os.environ.get('CPU_ONLY_TEST'))
    if not _CPU_ONLY and not "-ll:gpu" in sys.argv:
      os.environ["REALM_DEFAULT_ARGS"] = "-ll:gpu 1"
    if not _FF_BUILD_DOCS and not _CPU_ONLY:
      from legion_top import (
          legion_canonical_python_main,
          legion_canonical_python_cleanup,
      )
      import atexit, sys, os
      # run from jupyter
      if "ipykernel_launcher.py" in sys.argv[0]:
        sys_argv = ["python", "dummy.py"]
        argv_dict = load_jupyter_config()
        for key, value in argv_dict.items():
          sys_argv.append(key)
          sys_argv.append(str(value))
      else:
        sys_argv = [
          "python",
        ] + sys.argv
      legion_canonical_python_main(sys_argv)
      atexit.register(legion_canonical_python_cleanup)
  else:
    print("Using Legion Python")

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