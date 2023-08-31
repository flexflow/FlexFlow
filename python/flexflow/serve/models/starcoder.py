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
        self.max_seq_len = 256
        self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.dropout_p = hf_config.attn_pdrop
        self.hidden_size = hf_config.n_embd
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.max_position_embeddings = hf_config.n_positions
        self.num_attention_heads = hf_config.n_head
        self.num_hidden_layers = hf_config.n_layer
        self.vocab_size = hf_config.vocab_size
        self.intermediate_size = hf_config.n_inner
        self.n_head_kv = 1 if hf_config.multi_query else hf_config.n_head


class FlexFlowSTARCODER(FlexFlowModel):
    def __init__(
        self,
        mode,
        generation_config,
        ffconfig,
        hf_config,
        data_type,
        max_batch_size=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
        weights_filepath="",
        tokenizer_filepath="",
    ):
        self.mode = mode
        self.generation_config = generation_config
        self.ffconfig = ffconfig
        self.max_batch_size = max_batch_size
        self.data_type = data_type
        self.starcoder_config = STARCODERConfig(hf_config)
        self.starcoder_config.max_seq_length = max_seq_length
        self.starcoder_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1

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
        if (
            self.starcoder_config.n_head_kv < self.ffconfig.tensor_parallelism_degree
            or self.starcoder_config.n_head_kv % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of k/v attention heads ({self.starcoder_config.n_head_kv}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )
            
        self.build_model()

    def build_model(self):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [self.starcoder_config.max_num_tokens, 1]
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
            name="transformer_wte_weight",
        )
        positional_embedding = ffmodel.embedding(
            position_tensor,
            self.starcoder_config.max_position_embeddings,
            self.starcoder_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="transformer_wpe_weight",
        )

        hidden_states = ffmodel.add(token, positional_embedding)

        axes = [
            0,
        ]

        for i in range(self.starcoder_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)
            ln_1 = ffmodel.layer_norm(
                hidden_states,
                axes,
                True,
                self.starcoder_config.layer_norm_epsilon,
                name=f"layers_{i}_ln_1_weight",
            )

            assert self.mode == InferenceMode.INC_DECODING_MODE
            mha = ffmodel.inc_multiquery_self_attention(
                ln_1,
                self.starcoder_config.hidden_size,
                self.starcoder_config.num_attention_heads,
                self.starcoder_config.n_head_kv,
                self.starcoder_config.hidden_size
                // self.starcoder_config.num_attention_heads,
                self.starcoder_config.hidden_size
                // self.starcoder_config.num_attention_heads,
                0.0,  # dropout
                True,  # bias
                False,  # add_bias_kv
                False,  # add_zero_attn
                DataType.DT_NONE,  # data_type
                None,  # kernel initializer
                False,  # apply_rotary_embedding
                name=f"layers_{i}_attention_weight",
            )

            residual = ffmodel.add(mha, hidden_states)

            l2_norm = ffmodel.layer_norm(
                residual,
                axes,
                True,
                self.starcoder_config.layer_norm_epsilon,
                name=f"layers_{i}_ln_2_weight",
            )

            # mlp

            c_fc = ffmodel.dense(
                l2_norm,
                self.starcoder_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers_{i}_mlp_c_fc_weight",
            )
            activation = ffmodel.gelu(c_fc, False)
            c_proj = ffmodel.dense(
                activation,
                self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers_{i}_mlp_c_proj_weight",
            )
            hidden_states = ffmodel.add(residual, c_proj)

        ln_f = ffmodel.layer_norm(
            hidden_states,
            axes,
            True,
            self.starcoder_config.layer_norm_epsilon,
            name=f"transformer_ln_f_weight",
        )
        lm_head = ffmodel.dense(
            ln_f,
            self.starcoder_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head_weight",
        )

        if self.generation_config.do_sample:
            dense = ffmodel.scalar_true_divide(
                lm_head, self.generation_config.temperature, False
            )
            softmax = ffmodel.softmax(dense, -1)
            output = ffmodel.sampling(softmax, self.generation_config.topp)
        else:
            output = ffmodel.argmax(lm_head, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = name.replace("transformer.h", "layers").replace(".", "_")
            if "c_attn_weight" in name:
                name_q = name.replace("attn_c_attn", "attention_wq")
                name_k = name.replace("attn_c_attn", "attention_wk")
                name_v = name.replace("attn_c_attn", "attention_wv")
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
            elif "c_attn_bias" in name:
                name_q = name.replace("attn_c_attn", "attention_wq")
                name_k = name.replace("attn_c_attn", "attention_wk")
                name_v = name.replace("attn_c_attn", "attention_wv")
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
            elif "c_proj_bias" in name:
                name = name.replace("attn_c_proj", "attention_wo")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            elif "c_proj_weight" in name:
                name = name.replace("attn_c_proj", "attention_wo")
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
        model.lm_head.weight.detach().cpu().numpy().tofile(
            os.path.join(dst_folder, "lm_head_weight")
        )

    def get_layers_with_weights(self):
        layer_names = [
            "transformer_wte_weight",
            "transformer_wpe_weight",
            "transformer_ln_f_weight",
            "lm_head_weight",
        ] + [
            expr
            for i in range(self.starcoder_config.num_hidden_layers)
            for expr in (
                f"layers_{i}_ln_1_weight",
                f"layers_{i}_attention_weight",
                f"layers_{i}_ln_2_weight",
                f"layers_{i}_mlp_c_fc_weight",
                f"layers_{i}_mlp_c_proj_weight",
            )
        ]
        layers_with_weights = {
            layer_name: self.ffmodel.get_layer_by_name(layer_name)
            for layer_name in layer_names
        }

        return layers_with_weights
