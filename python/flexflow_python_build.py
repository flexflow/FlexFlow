#!/usr/bin/env python3

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

import argparse
import os
import sys
import stat

parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', required=True)
args = parser.parse_args()
build_dir = args.build_dir
if not os.path.isdir(build_dir):
    print(f"Error: Build directory {build_dir} does not exist")
    sys.exit(1)
build_dir = os.path.abspath(build_dir)
script_dir = os.path.abspath(os.path.dirname(__file__))
if not os.path.isdir(build_dir):
    print(f"Folder {build_dir} does not exist")
    sys.exit(1)
if not os.path.isdir(script_dir):
    print(f"Folder {script_dir} does not exist")
    sys.exit(1)
# Build flexflow_python script
flexflow_python_path = os.path.join(build_dir, "flexflow_python")
flexflow_python_path = os.path.abspath(flexflow_python_path)
lines = [
    '#! /usr/bin/env bash',
    f'BUILD_FOLDER="{build_dir}"',
    'SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]%/*}")"',
    'legion_python_args=("$@" "-ll:py" "1")',
    'if [[ "$SCRIPT_DIR" -ef "$BUILD_FOLDER" ]]; then',
    f'\tPYTHON_FOLDER="{script_dir}"',
    '\tPYLIB_PATH="$("$PYTHON_FOLDER"/flexflow/findpylib.py)"',
    '\tPYLIB_DIR="$(dirname "$PYLIB_PATH")"',
    '\texport LD_LIBRARY_PATH="$BUILD_FOLDER:$BUILD_FOLDER/deps/legion/lib:$PYLIB_DIR:$LD_LIBRARY_PATH"',
    '\texport PYTHONPATH="$PYTHON_FOLDER:$BUILD_FOLDER/deps/legion/bindings/python:$PYTHONPATH"',
    '\t$BUILD_FOLDER/deps/legion/bin/legion_python "${legion_python_args[@]}"',
    'else',
    '\tPYLIB_PATH="$(python3 -m flexflow.findpylib)"',
    '\tPYLIB_DIR="$(dirname "$PYLIB_PATH")"',
    '\texport LD_LIBRARY_PATH="$PYLIB_DIR:$LD_LIBRARY_PATH"',
    '\tlegion_python "${legion_python_args[@]}"',
    'fi'
]
with open(flexflow_python_path, "w+") as flexflow_python_file:
    for line in lines:
        flexflow_python_file.write(line + "\n")
cur_stat = os.stat(flexflow_python_path)
os.chmod(flexflow_python_path, cur_stat.st_mode | stat.S_IEXEC)

# Build set_python_envs.sh
python_envs_path = os.path.join(build_dir, "set_python_envs.sh")
python_envs_path = os.path.abspath(python_envs_path)
lines = [
    '#! /usr/bin/env bash',
    f'BUILD_FOLDER="{build_dir}"',
    f'PYTHON_FOLDER="{script_dir}"',
    'PYLIB_PATH="$("$PYTHON_FOLDER"/flexflow/findpylib.py)"',
    'PYLIB_DIR="$(dirname "$PYLIB_PATH")"',
    'export LD_LIBRARY_PATH="$BUILD_FOLDER:$BUILD_FOLDER/deps/legion/lib:$PYLIB_DIR:$LD_LIBRARY_PATH"',
    'export PYTHONPATH="$PYTHON_FOLDER:$BUILD_FOLDER/deps/legion/bindings/python:$PYTHONPATH"',
]
with open(python_envs_path, "w+") as python_envs_file:
    for line in lines:
        python_envs_file.write(line + "\n")
cur_stat = os.stat(python_envs_path)
os.chmod(python_envs_path, cur_stat.st_mode | stat.S_IEXEC)
