#!/usr/bin/env python

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

def run_flexflow(nodes, cpus, gpus, utility, fbmem, zcmem, opts):
  # Start building our command line for the subprocess invocation
  cmd_env = dict(os.environ.items())
  
  flexflow_python_path, flexflow_lib_dir, flexflow_lib64_dir = find_flexflow_python_exe()
  # print(flexflow_python_path, flexflow_lib_dir, flexflow_lib64_dir)
  
  if 'LD_LIBRARY_PATH' in cmd_env:
    cmd_env['LD_LIBRARY_PATH'] += ':' + flexflow_lib_dir + ':' + flexflow_lib64_dir
  else:
    cmd_env['LD_LIBRARY_PATH'] = flexflow_lib_dir + ':' + flexflow_lib64_dir

  # print(cmd_env['LD_LIBRARY_PATH'])

  cmd = [str(flexflow_python_path)]

  if cpus != 1:
    cmd += ['-ll:cpu', str(cpus)]
  if gpus > 0:
    cmd += ['-ll:gpu', str(gpus)]
    cmd += ['-ll:fsize', str(fbmem), '-ll:zsize', str(zcmem)]
  if utility != 1:
    cmd += ['-ll:util', str(utility)]

  # Now we can append the result of the flags to the command
  if opts:
    cmd += opts

  # print(cmd)
  # Launch the child process
  child_proc = subprocess.Popen(cmd, env = cmd_env)
  # Wait for it to finish running
  result = child_proc.wait()
  return result

def flexflow_driver():
  parser = argparse.ArgumentParser(
    description='FlexFlow Driver.')
  parser.add_argument(
          '--nodes', type=int, default=1, dest='nodes', help='Number of nodes to use')
  parser.add_argument(
          '--cpus', type=int, default=1, dest='cpus', help='Number of CPUs per node to use')
  parser.add_argument(
          '--gpus', type=int, default=1, dest='gpus', help='Number of GPUs per node to use')
  parser.add_argument(
          '--utility', type=int, default=1, dest='utility', help='Number of Utility processors per node to request for meta-work')
  parser.add_argument(
          '--fbmem', type=int, default=2048, dest='fbmem', help='Amount of framebuffer memory per GPU (in MBs)')
  parser.add_argument(
          '--zcmem', type=int, default=12192, dest='zcmem', help='Amount of zero-copy memory per node (in MBs)')
  args, opts = parser.parse_known_args()
  # See if we have at least one script file to run
  console = True
  for opt in opts:
    if '.py' in opt:
      console = False
      break
  assert console == False, "Please provide a python file"
  return run_flexflow(args.nodes, args.cpus, args.gpus, args.utility, args.fbmem, args.zcmem, opts)