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

import json
from flexflow.config import flexflow_dir

_CONFIG_FILENAME = None

def set_jupyter_config(filename):
    global _CONFIG_FILENAME
    _CONFIG_FILENAME = filename
    print("config file is set to:", _CONFIG_FILENAME)

def load_jupyter_config():
  cmd_dict_key = ["cpus", "gpus", "utility", "sysmem", "fbmem", "zcmem"]
  argv_dict = {}
  global _CONFIG_FILENAME
  if _CONFIG_FILENAME is None:
      raise Exception("Sorry, jupyter configuration file is not set, please call set_jupyter_config to set the path to the configuration json file.")
  with open(_CONFIG_FILENAME) as json_file:
        cmd_dict = json.load(json_file)
        for key in cmd_dict_key:
            if key in cmd_dict and cmd_dict[key]["value"] is not None:
               argv_dict[cmd_dict[key]["cmd"]] = cmd_dict[key]["value"]
  return argv_dict
