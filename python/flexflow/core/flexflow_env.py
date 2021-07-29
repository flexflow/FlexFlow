import os

# get python binding from env
_FF_PYTHON_BINDING = 'pybind11'

if 'FF_USE_CFFI' in os.environ:
  use_pybind = not int(os.environ['FF_USE_CFFI'])
else:
  use_pybind = True

if use_pybind:
  _FF_PYTHON_BINDING = 'pybind11'
else:
  _FF_PYTHON_BINDING = 'cffi'
  
# get python interpreter from env
_FF_PYTHON_INTERPRETER = 'legion'

if 'FF_USE_NATIVE_PYTHON' not in os.environ:
  use_native_python = 0
else:
  use_native_python = int(os.environ['FF_USE_NATIVE_PYTHON'])
  
if use_native_python:
  _FF_PYTHON_INTERPRETER = 'native'
else:
  _FF_PYTHON_INTERPRETER = 'legion'
  
def flexflow_python_binding():
  return _FF_PYTHON_BINDING
  
def flexflow_python_interpreter():
  return _FF_PYTHON_INTERPRETER