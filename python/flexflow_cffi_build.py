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
import subprocess

def find_flexflow_header(ffhome_dir):
    def try_prefix(prefix_dir):
        flexflow_ch_path = os.path.join(prefix_dir, 'include/flexflow', 'flexflow_c.h')
        flexflow_cxxh_path = os.path.join(prefix_dir, 'include/flexflow', 'model.h')
        if os.path.exists(flexflow_ch_path) and os.path.exists(flexflow_cxxh_path):
            flexflow_cxxh_dir = os.path.join(prefix_dir, 'include')
            return flexflow_cxxh_dir, flexflow_ch_path

    result = try_prefix(ffhome_dir)
    if result:
        return result

    raise Exception('Unable to locate flexflow_c.h and flexflow.h header file')

def build(output_dir, ffhome_dir):
    flexflow_cxxh_dir, flexflow_ch_path = find_flexflow_header(ffhome_dir)

    header = subprocess.check_output(['gcc', '-I', flexflow_cxxh_dir, '-E', '-P', flexflow_ch_path]).decode('utf-8')

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'flexflow_cffi_header.py.in')) as f:
        content = f.read()
    content = content.format(header=repr(header))

    if output_dir is None:
        output_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(output_dir, 'flexflow_cffi_header.py'), 'wb') as f:
        f.write(content.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffhome-dir', required=True)
    parser.add_argument('--output-dir', required=False)
    args = parser.parse_args()

    build(args.output_dir, args.ffhome_dir)
