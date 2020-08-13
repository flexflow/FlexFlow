#!/usr/bin/env python

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

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import subprocess

def find_legion_header(runtime_dir):
    def try_prefix(prefix_dir):
        legion_h_path = os.path.join(prefix_dir, 'legion.h')
        if os.path.exists(legion_h_path):
            return prefix_dir, legion_h_path

    result = try_prefix(runtime_dir)
    if result:
        return result

    raise Exception('Unable to locate legion.h header file')

def build(defines_dir, output_dir, runtime_dir):
    prefix_dir, legion_h_path = find_legion_header(runtime_dir)

    if defines_dir is not None:
        # For CMake, need to be told where the defines directory is:
        build_flags = ['-I', defines_dir]
    else:
        # For Make, legion_defines.h is in the source directory:
        build_flags = ['-I', os.path.dirname(os.path.realpath(__file__))]

    header = subprocess.check_output(['gcc', '-I', prefix_dir] + build_flags + ['-DLEGION_USE_PYTHON_CFFI', '-E', '-P', legion_h_path]).decode('utf-8')

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'legion_cffi.py.in')) as f:
        content = f.read()
    content = content.format(header=repr(header))

    if output_dir is None:
        output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(output_dir, 'legion_cffi.py'), 'wb') as f:
        f.write(content.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-dir', required=True)
    parser.add_argument('--defines-dir', required=False)
    parser.add_argument('--output-dir', required=False)
    args = parser.parse_args()

    build(args.defines_dir, args.output_dir, args.runtime_dir)
