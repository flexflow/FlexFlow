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
from distutils import sysconfig
from flexflow.findpylib import find_libpython

# When installing FlexFlow with pip, the library files are installed within
# the pip package folder, instead of at /usr/local/lib
packages_dir = sysconfig.get_python_lib(plat_specific=False, standard_lib=False)
ff_lib_path = os.path.join(packages_dir, "flexflow", "lib")
ld_lib_path = os.environ.get("LD_LIBRARY_PATH") or ""
# If the library exists at the ff_lib_path, rerun with the ff_lib_path in the LD_LIBRARY_PATH
rerun=False
if os.path.isdir(ff_lib_path) and ff_lib_path not in ld_lib_path.split(":"):
  os.environ["LD_LIBRARY_PATH"] = ff_lib_path + ":" + ld_lib_path
  rerun=True
pythonlib_folder = os.path.dirname(find_libpython() or "")
if os.path.isdir(pythonlib_folder) and pythonlib_folder not in ld_lib_path.split(":"):
  os.environ["LD_LIBRARY_PATH"] = pythonlib_folder + ":" + ld_lib_path
  rerun=True
if rerun:
  os.execv(sys.executable, ["python"] + sys.argv)

if flexflow_init_import():
  from legion_cffi import ffi, is_legion_python
  from .flexflowlib import flexflow_library
  
  # Default python mode
  if is_legion_python == False:
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