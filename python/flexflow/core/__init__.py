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

def rerun_if_needed():
  def update_ld_library_path_if_needed(path):
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH") or ""
    if path not in ld_lib_path.split(":"):
      os.environ["LD_LIBRARY_PATH"] = path + ":" + ld_lib_path
      return True
    return False
  from distutils import sysconfig
  # When installing FlexFlow with pip, the library files are installed within
  # the pip package folder, instead of at /usr/local/lib
  packages_dir = sysconfig.get_python_lib(plat_specific=False, standard_lib=False)
  ff_lib_path = os.path.join(packages_dir, "flexflow", "lib")
  # If the library exists at the ff_lib_path, rerun with the ff_lib_path in the LD_LIBRARY_PATH
  rerun=False
  if os.path.isdir(ff_lib_path):
    rerun = update_ld_library_path_if_needed(ff_lib_path)
  if rerun:
    run_from_python_c = ((sys.argv or [''])[0] == '-c')
    # re-running with os.execv only works with 'python -c' for python >= 3.10
    # (see https://bugs.python.org/issue23427)
    if not run_from_python_c:
      os.execv(sys.executable, ["python"] + sys.argv)
    elif hasattr(sys, 'orig_argv') and len(sys.argv) >= 3:
      py_args = sys.argv[2:]
      os.execv(sys.executable, ["python"] + py_args)

if flexflow_init_import():
  from legion_cffi import ffi, is_legion_python
  from .flexflowlib import flexflow_library
  
  # Default python mode
  if is_legion_python == False:
    os.environ["REALM_DEFAULT_ARGS"] = "-ll:gpu 1"
    rerun_if_needed()
    print("Using Default Python")
    from legion_top import (
        legion_canonical_python_main,
        legion_canonical_python_cleanup,
    )
    import atexit, sys, os
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