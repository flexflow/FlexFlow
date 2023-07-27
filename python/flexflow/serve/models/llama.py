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
import random

class LLAMAConfig:
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

class FlexFlowLLAMA:
    def __init__(self, mode, sampling_config, ffconfig, max_batch_size=1, max_seq_length=256, max_tokens_per_batch=64, use_full_precision=False):
        self.mode = mode
        self.sampling_config = sampling_config
        self.ffconfig = ffconfig
        self.max_batch_size = max_batch_size
        self.use_full_precision = use_full_precision
        self.llama_config = LLAMAConfig()
        self.llama_config.max_seq_length = max_seq_length
        self.llama_config.max_num_tokens = max_tokens_per_batch
        
        self.build_model()
    
    def build_model(self):
        ffmodel = FFModel(self.ffconfig)
        
        tokens_dims = [self.llama_config.max_num_tokens, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, 2**31-1), 0, 0)
        token = ffmodel.embedding(input_tensor, self.llama_config.vocab_size, self.llama_config.dim, AggrMode.AGGR_MODE_NONE, DataType.DT_FLOAT if self.use_full_precision else DataType.DT_HALF, None, embed_init)

        for i in range(self.llama_config.n_layers):
            ffmodel.set_transformer_layer_id(i)

            attn_norm = ffmodel.rms_norm(token, self.llama_config.norm_eps, self.llama_config.dim)

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multihead_attention(
                    attn_norm, 
                    self.llama_config.dim,
                    self.llama_config.n_heads,
                    self.llama_config.dim // self.llama_config.n_heads,
                    self.llama_config.dim // self.llama_config.n_heads,
                    0.0,    # dropout
                    False,  # bias 
                    False,  # add_bias_kv 
                    False,  # add_zero_attn 
                    None,   # kernel initializer
                    True    # apply_rotary_embedding 
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multihead_self_attention_verify(
                    attn_norm, 
                    self.llama_config.dim,
                    self.llama_config.n_heads,
                    self.llama_config.dim // self.llama_config.n_heads,
                    self.llama_config.dim // self.llama_config.n_heads,
                    0.0,    # dropout
                    False,  # bias 
                    False,  # add_bias_kv 
                    False,  # add_zero_attn 
                    None,   # kernel initializer
                    True    # apply_rotary_embedding 
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multihead_attention(
                    attn_norm, 
                    self.llama_config.dim,
                    self.llama_config.n_heads,
                    self.llama_config.dim // self.llama_config.n_heads,
                    self.llama_config.dim // self.llama_config.n_heads,
                    0.0,    # dropout
                    False,  # bias 
                    False,  # add_bias_kv 
                    False,  # add_zero_attn 
                    None,   # kernel initializer
                    True    # apply_rotary_embedding 
                )
            else:
                assert(False)
            
            token = ffmodel.add(token, mha)
            ff_norm = ffmodel.rms_norm(token, self.llama_config.norm_eps, self.llama_config.dim)
            w1 = ffmodel.dense(ff_norm, self.llama_config.hidden_dim, ActiMode.AC_MODE_NONE, False)
            w3 = ffmodel.dense(ff_norm, self.llama_config.hidden_dim, ActiMode.AC_MODE_NONE, False)
            sigmoid = ffmodel.sigmoid(w1)
            silu = ffmodel.multiply(w1, sigmoid)
            multi = ffmodel.multiply(silu, w3)
            w2 = ffmodel.dense(multi, self.llama_config.dim, ActiMode.AC_MODE_NONE, False)
            token = ffmodel.add(token, w2)

            token = ffmodel.rms_norm(token, self.llama_config.norm_eps, self.llama_config.dim)
            dense = ffmodel.dense(token, self.llama_config.vocab_size, ActiMode.AC_MODE_NONE, False)
            
            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                softmax = ffmodel.softmax(dense, -1)
                #output = ffmodel.beam_top_k(softmax, self.llama_config.max_beam_width, False)
                output = ffmodel.argmax(softmax, True)
            else:
                if self.sampling_config.do_sample:
                    dense = ffmodel.scalar_true_divide(dense, self.sampling_config.temperature, False)
                    softmax = ffmodel.softmax(dense, -1)
                    output = ffmodel.sampling(softmax, self.sampling_config.topp)
                else:
                    #output = ffmodel.arg_top_k(dense, 1, False)
                    output = ffmodel.argmax(dense, False)


