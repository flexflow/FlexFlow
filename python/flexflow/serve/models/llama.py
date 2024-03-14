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
import random


class LLAMAConfig:
    def __init__(self, hf_config):
        # self.max_seq_len = 256
        # self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 20
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.rms_norm_eps = hf_config.rms_norm_eps
        self.intermediate_size = hf_config.intermediate_size
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = (
            hf_config.num_attention_heads
            if hf_config.num_key_value_heads is None
            else hf_config.num_key_value_heads
        )


class FlexFlowLLAMA(FlexFlowModel):
    def __init__(
        self,
        mode,
        generation_config,
        ffconfig,
        hf_config,
        data_type,
        # max_batch_size=1,
        # max_seq_length=256,
        max_tokens_per_batch,
        weights_filepath="",
        tokenizer_filepath="",
    ):
        self.mode = mode
        self.generation_config = generation_config
        self.ffconfig = ffconfig
        # self.max_batch_size = max_batch_size
        self.data_type = data_type
        self.llama_config = LLAMAConfig(hf_config)
        # self.llama_config.max_seq_length = max_seq_length
        # self.llama_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = (
            max_tokens_per_batch + self.llama_config.max_spec_tree_token_num
        )

        # Sanity checks
        if self.llama_config.hidden_size % self.llama_config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.llama_config.hidden_size}) is not divisible by number of attention heads ({self.llama_config.num_attention_heads})"
            )

        # Sanity checks
        if (
            self.llama_config.num_attention_heads
            < self.ffconfig.tensor_parallelism_degree
            or self.llama_config.num_attention_heads
            % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.llama_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )

        self.build_model(
            max_tokens_per_batch
            if self.mode == InferenceMode.INC_DECODING_MODE
            else max_verify_tokens_per_batch
        )

    def build_model(self, max_tokens_per_batch):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [max_tokens_per_batch, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = ffmodel.embedding(
            input_tensor,
            self.llama_config.vocab_size,
            self.llama_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="tok_embeddings",
        )

        for i in range(self.llama_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                attn_norm = ffmodel.rms_norm(
                    token,
                    self.llama_config.rms_norm_eps,
                    self.llama_config.hidden_size,
                    name=f"layers_{i}_attention_norm",
                )
            else:
                token, attn_norm = ffmodel.residual_rms_norm(
                    token,
                    w2,
                    self.llama_config.rms_norm_eps,
                    self.llama_config.hidden_size,
                    name=f"layers_{i}_attention_norm",
                )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multiquery_self_attention(
                    attn_norm,
                    self.llama_config.hidden_size,
                    self.llama_config.num_attention_heads,
                    self.llama_config.num_key_value_heads,
                    self.llama_config.hidden_size
                    // self.llama_config.num_attention_heads,
                    self.llama_config.hidden_size
                    // self.llama_config.num_attention_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multiquery_self_attention_verify(
                    attn_norm,
                    self.llama_config.hidden_size,
                    self.llama_config.num_attention_heads,
                    self.llama_config.num_key_value_heads,
                    self.llama_config.hidden_size
                    // self.llama_config.num_attention_heads,
                    self.llama_config.hidden_size
                    // self.llama_config.num_attention_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multiquery_self_attention(
                    attn_norm,
                    self.llama_config.hidden_size,
                    self.llama_config.num_attention_heads,
                    self.llama_config.num_key_value_heads,
                    self.llama_config.hidden_size
                    // self.llama_config.num_attention_heads,
                    self.llama_config.hidden_size
                    // self.llama_config.num_attention_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention",
                )
            else:
                assert False

            token, ff_norm = ffmodel.residual_rms_norm(
                token,
                mha,
                self.llama_config.rms_norm_eps,
                self.llama_config.hidden_size,
                name=f"layers_{i}_ffn_norm",
            )
            w1 = ffmodel.dense(
                ff_norm,
                self.llama_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w1",
            )
            w3 = ffmodel.dense(
                ff_norm,
                self.llama_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w3",
            )
            multi = ffmodel.sigmoid_silu_multi(w1, w3)
            w2 = ffmodel.dense(
                multi,
                self.llama_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w2",
            )

        _, token = ffmodel.residual_rms_norm(
            token,
            w2,
            self.llama_config.rms_norm_eps,
            self.llama_config.hidden_size,
            name="norm",
        )
        dense = ffmodel.dense(
            token,
            self.llama_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="output",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(dense, -1)
            # output = ffmodel.beam_top_k(softmax, self.llama_config.max_beam_width, False)
            output = ffmodel.argmax(softmax, True)
        else:
            if self.generation_config.do_sample:
                dense = ffmodel.scalar_true_divide(
                    dense, self.generation_config.temperature, False
                )
                softmax = ffmodel.softmax(dense, -1)
                output = ffmodel.sampling(softmax, self.generation_config.topp)
            else:
                # output = ffmodel.arg_top_k(dense, 1, False)
                output = ffmodel.argmax(dense, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = (
                name.replace(".", "_")
                .replace("self_attn", "attention")
                .replace("q_proj", "wq")
                .replace("k_proj", "wk")
                .replace("v_proj", "wv")
                .replace("o_proj", "wo")
                .replace("mlp", "feed_forward")
                .replace("gate_proj", "w1")
                .replace("down_proj", "w2")
                .replace("up_proj", "w3")
                .replace("input_layernorm", "attention_norm")
                .replace("post_attention_layernorm", "ffn_norm")
                .replace("embed_tokens", "tok_embeddings")
                .replace("lm_head", "output")
                .replace("model_", "")
            )
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
