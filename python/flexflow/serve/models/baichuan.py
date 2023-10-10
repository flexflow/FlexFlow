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
from .base import FlexFlowModel
import random, torch, shutil

class BAICHUANConfig:
    def __init__(self, hf_config):
        print(f"hf_config: {hf_config}")
        self.max_beam_width = 1
        self.max_beam_depth = 8 
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.rms_norm_eps = hf_config.rms_norm_eps
        self.intermediate_size = hf_config.intermediate_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.num_key_value_heads = hf_config.num_attention_heads

class FlexFlowBAICHUAN(FlexFlowModel):
    def __init__(
        self,
        mode,
        generation_config,
        ffconfig,
        hf_config,
        data_type,
        max_tokens_per_batch,
        weights_filepath="",
        tokenizer_filepath=""
    ):
        self.mode = mode 
        self.generation_config = generation_config
        self.ffconfig = ffconfig
        self.data_type = data_type
        self.baichuan_config = BAICHUANConfig(hf_config)
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2 ** 31 -1

        # Sanity checks
        if self.baichuan_config.hidden_size % self.baichuan_config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.baichuan_config.hidden_size}) is not divisible by n_head ({self.baichuan_config.num_attention_heads})"
            )

        # Sanity checks
        if (
            self.baichuan_config.num_attention_heads < self.ffconfig.tensor_parallelism_degree
            or self.baichuan_config.num_attention_heads % self.ffconfig.tensor_parallelism_degree != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.baichuan_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )
        self.build_model(max_tokens_per_batch)

    def build_model(self, max_tokens_per_batch):
        ffmodel = FFModel(self.ffconfig)

        token_dims = [max_tokens_per_batch, 1]
        input_tensor = ffmodel.create_tensor(token_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)

        token = ffmodel.embedding(
            input_tensor,
            self.baichuan_config.vocab_size,
            self.baichuan_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name = "token_embedding",
        )

        for i in range(self.baichuan_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                attn_norm = ffmodel.rms_norm(
                    token,
                    self.baichuan_config.rms_norm_eps,
                    self.baichuan_config.hidden_size,
                    name = f"layers_{i}_attention_norm",
                )

            else:
                token, attn_norm = ffmodel.residual_rms_norm(
                    token,
                    w2,
                    self.baichuan_config.rms_norm_eps,
                    self.baichuan_config.hidden_size,
                name=f"layers_{i}_attention_norm",
                )     

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.inc_multihead_self_attention(
                    attn_norm,
                    self.baichuan_config.hidden_size,
                    self.baichuan_config.num_attention_heads,
                    self.baichuan_config.hidden_size // self.baichuan_config.num_attention_heads,
                    self.baichuan_config.hidden_size // self.baichuan_config.num_attention_heads,   
                    0.0, #dropout
                    False, #bias 
                    False, #add_bias_kv
                    False, #add_zero_attn
                    DataType.DT_NONE, #data_type
                    None, #kernel_initializer
                    True, #apply_rotary_pos_emb
                    name=f"layers_{i}_attention",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multihead_self_attention_verify(
                    attn_norm,
                    self.baichuan_config.hidden_size,
                    self.baichuan_config.num_attention_heads,
                    self.baichuan_config.hidden_size // self.baichuan_config.num_attention_heads,
                    self.baichuan_config.hidden_size // self.baichuan_config.num_attention_heads,
                    0.0, #dropout
                    False, #bias
                    False, #add_bias_kv
                    False, #add_zero_attn
                    DataType.DT_NONE, #data_type
                    None, #kernel_initializer
                    True, #apply_rotary_pos_emb
                    name=f"layers_{i}_attention",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multihead_self_attention(
                    attn_norm,
                    self.baichuan_config.hidden_size,
                    self.baichuan_config.num_attention_heads,
                    self.baichuan_config.hidden_size // self.baichuan_config.num_attention_heads,
                    self.baichuan_config.hidden_size // self.baichuan_config.num_attention_heads,
                    0.0, #dropout
                    False, #bias
                    False, #add_bias_kv
                    False, #add_zero_attn
                    DataType.DT_NONE, #data_type
                    None, #kernel_initializer
                    True, #apply_rotary_pos_emb
                    name=f"layers_{i}_attention",
                )
            else:
                assert False, "Invalid mode"
            token, ff_norm = ffmodel.residual_rms_norm(
                token,
                mha,
                self.baichuan_config.rms_norm_eps,
                self.baichuan_config.hidden_size,
                name=f"layers_{i}_ffn_norm",
            )

            w1 = ffmodel.dense(
                ff_norm,
                self.baichuan_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w1",
            )

            w3 = ffmodel.dense(
                ff_norm,
                self.baichuan_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False ,
                name=f"layers_{i}_feed_forward_w3",
            )

            multi = ffmodel.sigmoid_silu_multi(w1, w3)

            w2 = ffmodel.dense(
                multi,
                self.baichuan_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w2",
            )

        token = ffmodel.rms_norm(
            token,
            self.baichuan_config.rms_norm_eps,
            self.baichuan_config.hidden_size,
            name="norm",
        )

        dense = ffmodel.dense(
            token, 
            self.baichuan_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="output",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(dense, -1)
            #output = ffmodel.beam_top_k(softmax, self.llama_config.max_beam_width, False)
            output = ffmodel.argmax(softmax, True)
        else:
            if self.generation_config.do_sample:
                dense = ffmodel.scalar_true_divide(
                    dense, self.generation_config.temperature, False
                )
                softmax = ffmodel.softmax(dense, -1)
                output = ffmodel.sampling(softmax, self.generation_config.topp)
            else:
                #output = ffmodel.arg_top_k(dense, 1, False)
                output = ffmodel.argmax(dense, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        print(f"model: {model} and dst_folder: {dst_folder}")
        for name, params in model.named_parameters():
            name = (
                name.replace(".", "_")
                .replace("self_attn", "attention")
                .replace("W_pack", "Wqkv")
                .replace("o_proj", "wo")
                .replace("mlp", "feed_forward")
                .replace("gate_proj", "w1")
                .replace("down_proj", "w2")
                .replace("up_proj", "w3")
                .replace("input_layernorm", "attention_norm")
                .replace("post_attention_layernorm", "ffn_norm")
                .replace("embed_tokens", "token_embedding")
                .replace("lm_head", "output")
                .replace("model_", "")
            )
            print(f"0 name:{name} and params.shape: {params.shape} and")
            if "attention_Wqkv" in name:
                name_q = name.replace("attention_Wqkv", "attention_wq")
                name_k = name.replace("attention_Wqkv", "attention_wk")
                name_v = name.replace("attention_Wqkv", "attention_wv")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size,
                        model.config.hidden_size,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
                print(f"1 name:{name}  and name_q: {name_q} and name_k: {name_k} and name_v: {name_v}")
                print(f"2 q.shape: {q.shape} and k.shape: {k.shape} and v.shape: {v.shape}")
            else:
                print(f"3 name:{name} and params.shape: {params.shape} ")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))