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

from flexflow.core import *
import sys, random

class LLAMAConfig(dict):
    def __init__(self):
        self.n_layers = 32
        self.vocab_size = 3200
        self.n_heads = 32
        self.dim = 4096
        self.multiple_of = 256
        self.norm_eps = 1e-6
        self.total_requests = 2560
        self.incremental_mode = True
        self.hidden_dim = 11008
        self.max_seq_len = 256
        self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'LLAMAConfig' object has no attribute '{name}'")

class FlexFlowLLAMA:
    def __init__(self, max_batch_size=1, max_seq_length=256, max_tokens_per_batch=64, use_full_precision=False):
        self.max_batch_size = max_batch_size
        self.use_full_precision = use_full_precision
        self.llama_config = LLAMAConfig()
        self.llama_config.max_seq_length = max_seq_length
        self.llama_config.max_num_tokens = max_tokens_per_batch
        
        self.build_model()
    
    def build_model(self):
        ffconfig = FFConfig()
        ffmodel = FFModel(ffconfig)
        
        tokens_dims = [self.max_tokens_per_batch, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, sys.maxsize), 0, 0)
        token = ffmodel.embedding(input_tensor, self.llama_config.vocab_size, self.llama_config.dim, AggrMode.AGGR_MODE_NONE, DataType.DT_FLOAT if self.use_full_precision else DataType.DT_HALF, None, embed_init)

