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
import argparse, json, os, platform, subprocess, sys
from .utils import flexflow_dir

_version = sys.version_info

if _version.major == 3 and _version.minor >= 6: # Python 3.6 up:
  pass
else:
  raise Exception('Incompatible Python version')

os_name = platform.system()

if os_name == 'Linux':
  dylib_ext = '.so'
elif os_name == 'Darwin': # Don't currently support Darwin at the moment
  dylib_ext = '.dylib'
else:
  raise Exception('FlexFlow does not work on %s' % platform.system())
  
def find_flexflow_python_exe():
  ff_dir = flexflow_dir()
  
  # in pypi, flexflow_python is in flexflow/bin
  flexflow_python_path_1 = os.path.join(ff_dir, 'bin/flexflow_python')
  # with cmake install flexflow_python is in prefix/bin
  cmake_prefix_dir = os.path.abspath(os.path.join(ff_dir, os.pardir))
  flexflow_python_path_2 = os.path.join(cmake_prefix_dir, 'bin/flexflow_python')
  # with makefile or cmake build, flexflow is in ff_home/python
  python_dir = os.path.abspath(os.path.join(ff_dir, os.pardir))
  flexflow_python_path_3 = os.path.join(python_dir, 'flexflow_python')
  
  # print(ff_dir)
  # print(flexflow_python_path_1, flexflow_python_path_2, flexflow_python_path_3)
  if os.path.exists(flexflow_python_path_1):
    flexflow_lib_dir = os.path.join(ff_dir, 'lib')
    flexflow_lib64_dir = os.path.join(ff_dir, 'lib64')
    return flexflow_python_path_1, flexflow_lib_dir, flexflow_lib64_dir
  elif os.path.exists(flexflow_python_path_2):
    flexflow_lib_dir = os.path.join(cmake_prefix_dir, 'lib')
    flexflow_lib64_dir = os.path.join(cmake_prefix_dir, 'lib64')
    return flexflow_python_path_2, flexflow_lib_dir, flexflow_lib64_dir
  elif os.path.exists(flexflow_python_path_3):
    flexflow_lib_dir = python_dir
    flexflow_lib64_dir = python_dir
    return flexflow_python_path_3, flexflow_lib_dir, flexflow_lib64_dir
  else:
    raise Exception('Unable to locate flexflow_python')

def run_flexflow(freeze_on_error, backtrace, opts):
  flexflow_python_path, flexflow_lib_dir, flexflow_lib64_dir = find_flexflow_python_exe()
  # print(flexflow_python_path, flexflow_lib_dir, flexflow_lib64_dir)
  
  # set LD_LIBRARY_PATH  
  cmd_env = dict(os.environ.items())
  if 'LD_LIBRARY_PATH' in cmd_env:
    cmd_env['LD_LIBRARY_PATH'] += ':' + flexflow_lib_dir + ':' + flexflow_lib64_dir
  else:
    cmd_env['LD_LIBRARY_PATH'] = flexflow_lib_dir + ':' + flexflow_lib64_dir
    
  # freeze on error
  if freeze_on_error:
      cmd_env["LEGION_FREEZE_ON_ERROR"] = str(1)
      
  # print backtrace
  if backtrace:
      cmd_env["LEGION_BACKTRACE"] = str(1)

  # Start building our command line for the subprocess invocation
  cmd = [str(flexflow_python_path)]
  cmd += opts
  # print(cmd)
  
  # Launch the child process
  child_proc = subprocess.Popen(cmd, env = cmd_env)
  # Wait for it to finish running
  result = child_proc.wait()
  return result

def flexflow_driver():
  parser = argparse.ArgumentParser(description='FlexFlow Driver.')
  parser.add_argument(
    "--freeze-on-error",
    dest="freeze_on_error",
    action="store_true",
    required=False,
    help="if the program crashes, freeze execution right before exit so a debugger can be attached",
  )
  parser.add_argument(
    "--backtrace",
    dest="backtrace",
    action="store_true",
    required=False,
    help="if the program crashes, print the backtrace where an error occurs",
  )
  args, opts = parser.parse_known_args()
  # See if we have at least one script file to run
  console = True
  for opt in opts:
    if '.py' in opt:
      console = False
      break
  assert console == False, "Please provide a python file"
  return run_flexflow(args.freeze_on_error, args.backtrace, opts)