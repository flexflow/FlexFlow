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
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 20
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.rms_norm_eps = hf_config.rms_norm_eps
        self.intermediate_size = hf_config.intermediate_size
        self.rotary_embedding_meta = RotaryEmbeddingMeta(
            apply_rotary_embedding=True,
            rope_theta=hf_config.rope_theta if "rope_theta" in hf_config.__dict__ else 10000.0,
        )
        if "rope_scaling" in hf_config.__dict__:
            if hf_config.rope_scaling is not None:
                self.rotary_embedding_meta.rope_type = hf_config.rope_scaling["rope_type"]
                self.rotary_embedding_meta.factor = hf_config.rope_scaling["factor"]
                self.rotary_embedding_meta.low_freq_factor = hf_config.rope_scaling["low_freq_factor"]
                self.rotary_embedding_meta.high_freq_factor = hf_config.rope_scaling["high_freq_factor"]
                self.rotary_embedding_meta.original_max_position_embeddings = hf_config.rope_scaling["original_max_position_embeddings"]
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
        self.data_type = data_type
        self.llama_config = LLAMAConfig(hf_config)
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2 ** 31 - 1
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
            name="embed_tokens",
        )

        for i in range(self.llama_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                attn_norm = ffmodel.rms_norm(
                    token,
                    self.llama_config.rms_norm_eps,
                    self.llama_config.hidden_size,
                    name=f"layers.{i}.input_layernorm",
                )
            else:
                token, attn_norm = ffmodel.residual_rms_norm(
                    token,
                    w2,
                    self.llama_config.rms_norm_eps,
                    self.llama_config.hidden_size,
                    name=f"layers.{i}.input_layernorm",
                )

            qkv_proj = ffmodel.dense(
                attn_norm,
                3 * self.llama_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attn.qkv_proj",
            )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multiquery_self_attention(
                    qkv_proj,
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
                    self.llama_config.rotary_embedding_meta,
                    name=f"layers.{i}.self_attn",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multiquery_self_attention_verify(
                    qkv_proj,
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
                    self.llama_config.rotary_embedding_meta,
                    name=f"layers.{i}.self_attn",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multiquery_self_attention(
                    qkv_proj,
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
                    self.llama_config.rotary_embedding_meta,
                    name=f"layers.{i}.self_attn",
                )
            else:
                assert False

            o_proj = ffmodel.dense(
                mha,
                self.llama_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attn.o_proj"
            )

            token, ff_norm = ffmodel.residual_rms_norm(
                token,
                o_proj,
                self.llama_config.rms_norm_eps,
                self.llama_config.hidden_size,
                name=f"layers.{i}.post_attention_layernorm",
            )
            w1 = ffmodel.dense(
                ff_norm,
                self.llama_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.gate_proj",
            )
            w3 = ffmodel.dense(
                ff_norm,
                self.llama_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.up_proj",
            )
            multi = ffmodel.sigmoid_silu_multi(w1, w3)
            w2 = ffmodel.dense(
                multi,
                self.llama_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.down_proj",
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
            name="lm_head",
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
                softmax = ffmodel.softmax(dense, -1)
                output = ffmodel.argmax(softmax, False)

        self.ffmodel = ffmodel

    def convert_hf_weight_name(name):
        return name.replace("model.", "")

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = FlexFlowLLAMA.convert_hf_weight_name(name)
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
        # LM head weight
        model.lm_head.weight.detach().cpu().numpy().tofile(
            os.path.join(dst_folder, "lm_head.weight")
        )
