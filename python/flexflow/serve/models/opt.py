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
import random, shutil


class OPTConfig:
    def __init__(self, hf_config):
        self.max_seq_len = 256
        self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.do_layer_norm_before = hf_config.do_layer_norm_before
        self.dropout = hf_config.dropout
        self.enable_bias = hf_config.enable_bias
        self.ffn_dim = hf_config.ffn_dim
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_elementwise_affine = hf_config.layer_norm_elementwise_affine
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.word_embed_proj_dim = hf_config.word_embed_proj_dim


class FlexFlowOPT(FlexFlowModel):
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
        self.opt_config = OPTConfig(hf_config)
        self.opt_config.max_seq_length = max_seq_length
        self.opt_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1

        # Sanity checks
        if self.opt_config.hidden_size % self.opt_config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.opt_config.hidden_size}) is not divisible by n_head ({self.opt_config.num_attention_heads})"
            )

        # Sanity checks
        if (
            self.opt_config.num_attention_heads
            < self.ffconfig.tensor_parallelism_degree
            or self.opt_config.num_attention_heads
            % self.ffconfig.tensor_parallelism_degree
            != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.opt_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )

        self.build_model()

    def build_model(self):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [self.opt_config.max_num_tokens, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)
        position_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        # OPT model positional embedding start offset is 2
        ffmodel.set_position_offset(2)
        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = ffmodel.embedding(
            input_tensor,
            self.opt_config.vocab_size,
            self.opt_config.word_embed_proj_dim,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="embed_tokens_weight",
        )
        positional_embedding = ffmodel.embedding(
            position_tensor,
            self.opt_config.max_position_embeddings,
            self.opt_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="embed_positions_weight",
        )

        residual = ffmodel.add(token, positional_embedding)

        axes = [
            0,
        ]

        for i in range(self.opt_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            if self.opt_config.do_layer_norm_before:
                hidden_states = ffmodel.layer_norm(
                    residual,
                    axes,
                    self.opt_config.layer_norm_elementwise_affine,
                    1e-05,
                    name=f"layers_{i}_attention_layer_norm_weight",
                )
            else:
                hidden_states = residual

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multihead_self_attention(
                    hidden_states,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    0.0,  # dropout
                    True,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.opt_config.hidden_size / self.opt_config.num_attention_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers_{i}_attention_weight",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multihead_self_attention_verify(
                    hidden_states,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    0.0,  # dropout
                    True,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.opt_config.hidden_size / self.opt_config.num_attention_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers_{i}_attention_weight",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multihead_self_attention(
                    hidden_states,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    0.0,  # dropout
                    True,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.opt_config.hidden_size / self.opt_config.num_attention_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers_{i}_attention_weight",
                )
            else:
                assert False

            residual = ffmodel.add(mha, residual)

            # This is either a before or after attention LayerNorm. In both cases, we need to compute the LN here.
            norm_name = (
                f"layers_{i}_final_layer_norm_weight"
                if self.opt_config.do_layer_norm_before
                else f"layers_{i}_attention_layer_norm_weight"
            )
            ff_norm = ffmodel.layer_norm(
                residual,
                axes,
                self.opt_config.layer_norm_elementwise_affine,
                1e-05,
                name=norm_name,
            )

            if not self.opt_config.do_layer_norm_before:
                residual = ff_norm

            fc1 = ffmodel.dense(
                ff_norm,
                self.opt_config.ffn_dim,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers_{i}_fc1_weight",
            )
            activation = ffmodel.relu(fc1, False)
            fc2 = ffmodel.dense(
                activation,
                self.opt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers_{i}_fc2_weight",
            )
            residual = ffmodel.add(residual, fc2)

            if not self.opt_config.do_layer_norm_before:
                residual = ffmodel.layer_norm(
                    residual,
                    axes,
                    self.opt_config.layer_norm_elementwise_affine,
                    1e-05,
                    name=f"layers_{i}_final_layer_norm_weight",
                )

        all_final_norm = ffmodel.layer_norm(
            residual,
            axes,
            self.opt_config.layer_norm_elementwise_affine,
            1e-05,
            name=f"final_layer_norm_weight",
        )
        lm_head = ffmodel.dense(
            all_final_norm,
            self.opt_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="embed_tokens_weight_lm_head",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(lm_head, -1)
            # output = ffmodel.beam_top_k(softmax, self.opt_config.max_beam_width, False)
            output = ffmodel.argmax(softmax, True)
        else:
            if self.generation_config.do_sample:
                dense = ffmodel.scalar_true_divide(
                    lm_head, self.generation_config.temperature, False
                )
                softmax = ffmodel.softmax(dense, -1)
                output = ffmodel.sampling(softmax, self.generation_config.topp)
            else:
                # output = ffmodel.arg_top_k(lm_head, 1, False)
                output = ffmodel.argmax(lm_head, False)

        self.ffmodel = ffmodel

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = (
                name.replace(".", "_")
                .replace("decoder_", "")
                .replace("model_", "")
                .replace("self_attn", "attention")
                .replace("q_proj", "wq")
                .replace("k_proj", "wk")
                .replace("v_proj", "wv")
                .replace("out_proj", "wo")
            )
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
        # copy embedding weights
        shutil.copy(
            os.path.join(dst_folder, "embed_tokens_weight"),
            os.path.join(dst_folder, "embed_tokens_weight_lm_head"),
        )

    def get_layers_with_weights(self):
        layer_names = [
            "embed_tokens_weight",
            "embed_positions_weight",
            "final_layer_norm_weight",
            "embed_tokens_weight_lm_head",
        ] + [
            expr
            for i in range(self.opt_config.num_hidden_layers)
            for expr in (
                f"layers_{i}_attention_layer_norm_weight",
                f"layers_{i}_attention_weight",
                f"layers_{i}_final_layer_norm_weight",
                f"layers_{i}_fc1_weight",
                f"layers_{i}_fc2_weight",
            )
        ]
        layers_with_weights = {
            layer_name: self.ffmodel.get_layer_by_name(layer_name)
            for layer_name in layer_names
        }

        return layers_with_weights
