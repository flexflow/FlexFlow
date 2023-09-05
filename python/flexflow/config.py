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
_FF_PYTHON_BINDING = "cffi"

if "FF_USE_CFFI" in os.environ:
    use_pybind = not int(os.environ["FF_USE_CFFI"])
else:
    use_pybind = False

if use_pybind:
    _FF_PYTHON_BINDING = "pybind11"
else:
    _FF_PYTHON_BINDING = "cffi"


def flexflow_python_binding():
    return _FF_PYTHON_BINDING


_FF_ALREADY_INITIALIZED = False


def flexflow_already_initialized():
    global _FF_ALREADY_INITIALIZED
    return _FF_ALREADY_INITIALIZED


def set_flexflow_initialized():
    global _FF_ALREADY_INITIALIZED
    if _FF_ALREADY_INITIALIZED == True:
        raise RuntimeError(
            "Attempting to set _FF_ALREADY_INITIALIZED=True, but _FF_ALREADY_INITIALIZED is already True"
        )
    _FF_ALREADY_INITIALIZED = True


# FlexFlow dir
_FF_DIR = os.path.dirname(os.path.realpath(__file__))


def flexflow_dir():
    return _FF_DIR

# Get runtime configs from the command line 
def get_configs():
  import argparse,json
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-config-file",
    help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
    type=str,
    default=None,
  )
  args, unknown = parser.parse_known_args()
  if args.config_file is not None:
    with open(args.config_file) as f:
      return json.load(f)
  else:
    return None
