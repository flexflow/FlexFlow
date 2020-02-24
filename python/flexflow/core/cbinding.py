#!/usr/bin/env python

# Copyright 2020 Stanford University
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

import cffi
import subprocess

_flexflow_header = subprocess.check_output(['gcc', '-I/home/wwu12/FlexFlow/include', '-E', '-P', '/home/wwu12/FlexFlow/python/flexflow_c_dev.h']).decode('utf-8')
ffi = cffi.FFI()
ffi.cdef(_flexflow_header)
ffc = ffi.dlopen(None)

class FFConfig(object):
  def __init__(self):
    self.handle = ffc.flexflow_config_create()
    self._handle = ffi.gc(self.handle, ffc.flexflow_config_destroy)
    
  def parse_args(self):
    ffc.flexflow_config_parse_default_args(self.handle)
    
  def get_batch_size(self):
    return ffc.flexflow_config_get_batch_size(self.handle)
  
  def get_workers_per_node(self):
    return ffc.flexflow_config_get_workers_per_node(self.handle)
  
  def get_num_nodes(self):
    return ffc.flexflow_config_get_num_nodes(self.handle)
    
  def get_epochs(self):
    return ffc.flexflow_config_get_epochs(self.handle)
    
class FFModel(object):
  def __init__(self, config):
    self.handle = ffc.flexflow_model_create(config.handle)
    self._handle = ffi.gc(self.handle, ffc.flexflow_model_destroy)