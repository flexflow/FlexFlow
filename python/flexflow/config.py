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

import os

# python binding
_FF_PYTHON_BINDING = 'cffi'

if 'FF_USE_CFFI' in os.environ:
  use_pybind = not int(os.environ['FF_USE_CFFI'])
else:
  use_pybind = False

if use_pybind:
  _FF_PYTHON_BINDING = 'pybind11'
else:
  _FF_PYTHON_BINDING = 'cffi'
  
def flexflow_python_binding():
  return _FF_PYTHON_BINDING
  
# python interpreter
_FF_PYTHON_INTERPRETER = 'legion'

if 'FF_USE_NATIVE_PYTHON' not in os.environ:
  use_native_python = 0
else:
  use_native_python = int(os.environ['FF_USE_NATIVE_PYTHON'])
  
if use_native_python:
  _FF_PYTHON_INTERPRETER = 'native'
else:
  _FF_PYTHON_INTERPRETER = 'legion'
  
def flexflow_python_interpreter():
  return _FF_PYTHON_INTERPRETER

# build docs
_FF_BUILD_DOCS = bool(os.environ.get('READTHEDOCS') or os.environ.get("FF_BUILD_DOCS"))
  
# init import
# It is used to run __init__.py in flexflow/core
# The following cases __init__.py is not needed:
# 1. build docs = True
_FF_INIT_IMPORT = _FF_BUILD_DOCS == False

def flexflow_init_import():
  return _FF_INIT_IMPORT
  
# FlexFlow dir
_FF_DIR = os.path.dirname(os.path.realpath(__file__))

def flexflow_dir():
  return _FF_DIR
