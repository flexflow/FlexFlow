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
import random, torch


class STARCODERConfig:
    def __init__(self, hf_config):
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 20
        self.dropout_p = hf_config.attn_pdrop
        self.hidden_size = hf_config.n_embd
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.max_position_embeddings = hf_config.n_positions
        self.num_hidden_layers = hf_config.n_layer
        self.vocab_size = hf_config.vocab_size
        self.intermediate_size = hf_config.n_inner
        self.n_head_kv = 1 if hf_config.multi_query else hf_config.n_head
        self.rotary_embedding_meta = RotaryEmbeddingMeta(apply_rotary_embedding=False)
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.n_head
        self.num_key_value_heads = self.n_head_kv


class FlexFlowSTARCODER(FlexFlowModel):
    def __init__(
        self,
        mode,
        generation_config,
        ffconfig,
        hf_config,
        data_type,
        max_tokens_per_batch,
        weights_filepath="",
        tokenizer_filepath="",
    ):
        self.mode = mode
        self.generation_config = generation_config
        self.ffconfig = ffconfig
        self.data_type = data_type
        self.starcoder_config = STARCODERConfig(hf_config)
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = (
            max_tokens_per_batch + self.starcoder_config.max_spec_tree_token_num
        )

        # Sanity checks
        if (
            self.starcoder_config.hidden_size
            % self.starcoder_config.num_attention_heads
            != 0
        ):
            raise ValueError(
                f"Hidden size ({self.starcoder_config.hidden_size}) is not divisible by n_head ({self.starcoder_config.num_attention_heads})"
            )

        # Sanity checks
        if (
            self.starcoder_config.num_attention_heads
            < self.ffconfig.tensor_parallelism_degree
            or self.starcoder_config.num_attention_heads
            % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.starcoder_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
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
        position_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        ffmodel.set_position_offset(0)
        token = ffmodel.embedding(
            input_tensor,
            self.starcoder_config.vocab_size,
            self.starcoder_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="wte",
        )
        positional_embedding = ffmodel.embedding(
            position_tensor,
            self.starcoder_config.max_position_embeddings,
            self.starcoder_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="wpe",
        )

        axes = [
            0,
        ]

        for i in range(self.starcoder_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            hidden_states, ln_1 = ffmodel.residual_layer_norm(
                token if i == 0 else residual,
                positional_embedding if i == 0 else c_proj,
                None,
                False,
                axes,
                True,
                self.starcoder_config.layer_norm_epsilon,
                name=f"layers.{i}.ln_1",
            )

            qkv_proj = ffmodel.dense(
                ln_1,
                3 * self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attn.qkv_proj",
            )

            assert self.mode == InferenceMode.INC_DECODING_MODE
            o_proj = ffmodel.inc_multiquery_self_attention(
                qkv_proj,
                self.starcoder_config.hidden_size,
                self.starcoder_config.num_attention_heads,
                self.starcoder_config.n_head_kv,
                self.starcoder_config.hidden_size
                // self.starcoder_config.num_attention_heads,
                self.starcoder_config.hidden_size
                // self.starcoder_config.num_attention_heads,
                0.0,  # dropout
                True,  # qkv_bias
                False,  # final_bias
                False,  # add_zero_attn
                DataType.DT_NONE,  # data_type
                None,  # kernel initializer
                self.starcoder_config.rotary_embedding_meta,
                name=f"layers.{i}.attn.c_attn",
            )

            mha = ffmodel.dense(
                o_proj,
                self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.self_attn.o_proj"
            )

            residual, l2_norm = ffmodel.residual_layer_norm(
                hidden_states,
                mha,
                None,
                False,
                residual,
                axes,
                True,
                self.starcoder_config.layer_norm_epsilon,
                name=f"layers.{i}.ln_2",
            )

            # mlp

            c_fc = ffmodel.dense(
                l2_norm,
                self.starcoder_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.mlp.c_fc",
            )
            activation = ffmodel.gelu(c_fc, False)
            c_proj = ffmodel.dense(
                activation,
                self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.mlp.c_proj",
            )

        _, ln_f = ffmodel.residual_layer_norm(
            residual,
            c_proj,
            None,
            False,
            axes,
            True,
            self.starcoder_config.layer_norm_epsilon,
            name=f"ln_f",
        )
        lm_head = ffmodel.dense(
            ln_f,
            self.starcoder_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.generation_config.do_sample:
            dense = ffmodel.scalar_true_divide(
                lm_head, self.generation_config.temperature, False
            )
            softmax = ffmodel.softmax(dense, -1)
            output = ffmodel.sampling(softmax, self.generation_config.topp)
        else:
            softmax = ffmodel.softmax(lm_head, -1)
            output = ffmodel.argmax(softmax, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = name.replace("transformer.h", "layers").replace("transformer.", "")
            if "attn.c_attn.weight" in name:
                name_q = name.replace("attn.c_attn", "attn.c_attn.q_proj")
                name_k = name.replace("attn.c_attn", "attn.c_attn.k_proj")
                name_v = name.replace("attn.c_attn", "attn.c_attn.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // model.config.num_attention_heads,
                        model.config.hidden_size // model.config.num_attention_heads,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            elif "attn.c_attn.bias" in name:
                name_q = name.replace("attn.c_attn", "attn.c_attn.q_proj")
                name_k = name.replace("attn.c_attn", "attn.c_attn.k_proj")
                name_v = name.replace("attn.c_attn", "attn.c_attn.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // model.config.num_attention_heads,
                        model.config.hidden_size // model.config.num_attention_heads,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            elif "attn.c_proj.bias" in name:
                name = name.replace("attn.c_proj", "attn.c_attn.o_proj")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            elif "attn.c_proj.weight" in name:
                name = name.replace("attn.c_proj", "attn.c_attn.o_proj")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
        model.lm_head.weight.detach().cpu().numpy().tofile(
            os.path.join(dst_folder, "lm_head.weight")
        )
