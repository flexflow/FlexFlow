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
script_path = os.path.join(build_dir, "flexflow_python")
if not os.path.isdir(build_dir):
    print(f"Folder {build_dir} does not exist")
    sys.exit(1)
if not os.path.isdir(script_dir):
    print(f"Folder {script_dir} does not exist")
    sys.exit(1)
script_path = os.path.abspath(script_path)
lines = [
    '#! /usr/bin/env bash',
    f'BUILD_FOLDER="{build_dir}"',
    'SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]%/*}")"',
    'if [[ "$SCRIPT_DIR" == "$BUILD_FOLDER" ]]; then',
    f'\tPYTHON_FOLDER="{script_dir}"',
    '\tPYLIB_PATH="$("$PYTHON_FOLDER"/flexflow/findpylib.py)"',
    '\tPYLIB_DIR="$(dirname "$PYLIB_PATH")"',
    '\texport LD_LIBRARY_PATH="$BUILD_FOLDER:$BUILD_FOLDER/deps/legion/lib:$PYLIB_DIR:$LD_LIBRARY_PATH"',
    '\texport PYTHONPATH="$PYTHON_FOLDER:$BUILD_FOLDER/deps/legion/bindings/python:$PYTHONPATH"',
    '\t$BUILD_FOLDER/deps/legion/bin/legion_python "$@"',
    'else',
    '\tlegion_python "$@"',
    'fi'
]

with open(script_path, "w+") as script_file:
    for line in lines:
        script_file.write(line + "\n")

cur_stat = os.stat(script_path)
os.chmod(script_path, cur_stat.st_mode | stat.S_IEXEC)
