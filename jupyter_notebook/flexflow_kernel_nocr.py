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

from __future__ import print_function
from ipykernel.ipkernel import IPythonKernel
import sys

__version__ = '0.1'

class FlexFlowKernelNoCR(IPythonKernel):
    implementation = 'flexflow_kernel_nocr'
    implementation_version = __version__

    banner = "FlexFlow IPython Kernel for SM"
    language = 'python'
    language_version = __version__
    language_info = {'name': 'flexflow_kernel_nocr',
                     'mimetype': 'text/x-python',
                     'codemirror_mode': {
                        'name': 'ipython',
                        'version': 3
                        },
                     'pygments_lexer': 'ipython3',
                     'nbconvert_exporter': 'python',
                     'file_extension': '.py'}

    def __init__(self, **kwargs):
        self.__stdout = None
        self._set_stdout()
        print("Init FlexFlow kernel for single node or multi-nodes without control replication.")
        self._reset_stdout()
        super().__init__(**kwargs)

    def _set_stdout(self):
        assert(self.__stdout == None), "stdout should be None"
        self.__stdout = sys.stdout
        sys.stdout = open('/dev/stdout', 'w')

    def _reset_stdout(self):
        assert(self.__stdout != None), "stdout should not be None"
        sys.stdout = self.__stdout

if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=FlexFlowKernelNoCR)
