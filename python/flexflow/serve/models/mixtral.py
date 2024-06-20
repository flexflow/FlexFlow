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


class MixtralConfig:
    def __init__(self, hf_config):
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 64
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = hf_config.num_key_value_heads
        self.num_experts_per_tok = hf_config.num_experts_per_tok
        self.num_local_experts = hf_config.num_local_experts
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.output_router_logits = hf_config.output_router_logits
        self.rms_norm_eps = hf_config.rms_norm_eps
        self.rope_theta = hf_config.rope_theta
        self.router_aux_loss_coef = hf_config.router_aux_loss_coef
        self.sliding_window = hf_config.sliding_window
        self.tie_word_embeddings = hf_config.tie_word_embeddings
        self.vocab_size = hf_config.vocab_size


class FlexFlowMixtral(FlexFlowModel):
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
        self.mixtral_config = MixtralConfig(hf_config)
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = (
            max_tokens_per_batch + self.mixtral_config.max_spec_tree_token_num
        )

        # Sanity checks
        if self.mixtral_config.hidden_size % self.mixtral_config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.mixtral_config.hidden_size}) is not divisible by num_attention_heads ({self.mixtral_config.num_attention_heads})"
            )
        if (
            self.mixtral_config.num_attention_heads < self.ffconfig.tensor_parallelism_degree
            or self.mixtral_config.num_attention_heads % self.ffconfig.tensor_parallelism_degree != 0
        ):
            raise ValueError(
                f"Number of q attention heads ({self.mixtral_config.num_attention_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
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
            self.mixtral_config.vocab_size,
            self.mixtral_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="embed_tokens",
        )

        for i in range(self.mixtral_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                attn_norm = ffmodel.rms_norm(
                    token,
                    self.mixtral_config.rms_norm_eps,
                    self.mixtral_config.hidden_size,
                    name=f"layers_{i}_input_layernorm",
                )
            else:
                token, attn_norm = ffmodel.residual_rms_norm(
                    token,
                    mlp_out,
                    self.mixtral_config.rms_norm_eps,
                    self.mixtral_config.hidden_size,
                    name=f"layers_{i}_input_layernorm",
                )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multiquery_self_attention(
                    attn_norm,
                    self.mixtral_config.hidden_size,
                    self.mixtral_config.num_attention_heads,
                    self.mixtral_config.num_key_value_heads,
                    self.mixtral_config.hidden_size
                    // self.mixtral_config.num_attention_heads,
                    self.mixtral_config.hidden_size
                    // self.mixtral_config.num_attention_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_self_attn",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multiquery_self_attention_verify(
                    attn_norm,
                    self.mixtral_config.hidden_size,
                    self.mixtral_config.num_attention_heads,
                    self.mixtral_config.num_key_value_heads,
                    self.mixtral_config.hidden_size
                    // self.mixtral_config.num_attention_heads,
                    self.mixtral_config.hidden_size
                    // self.mixtral_config.num_attention_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_self_attn",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multiquery_self_attention(
                    attn_norm,
                    self.mixtral_config.hidden_size,
                    self.mixtral_config.num_attention_heads,
                    self.mixtral_config.num_key_value_heads,
                    self.mixtral_config.hidden_size
                    // self.mixtral_config.num_attention_heads,
                    self.mixtral_config.hidden_size
                    // self.mixtral_config.num_attention_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_self_attn",
                )
            else:
                assert False

            token, ff_norm = ffmodel.residual_rms_norm(
                token,
                mha,
                self.mixtral_config.rms_norm_eps,
                self.mixtral_config.hidden_size,
                name=f"layers_{i}_post_attention_layernorm",
            )
            ## Start of MoE block ##
            gate = ffmodel.dense(
                ff_norm,
                self.mixtral_config.num_local_experts,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_block_sparse_moe_gate",
            )
            gate = ffmodel.softmax(gate, name=f"layers_{i}_block_sparse_moe_softmax")
            topk_values, topk_indices = ffmodel.top_k(
                ff_norm,
                self.mixtral_config.num_experts_per_tok,
                False,
                name=f"layers_{i}_block_sparse_moe_topk",
            )
            grouped_tokens = ffmodel.group_by(
                ff_norm,
                topk_indices,
                self.mixtral_config.num_local_experts,
                name=f"layers_{i}_block_sparse_moe_groupby",
            )
            expert_predictions = []
            for expert_idx in range(self.mixtral_config.num_local_experts):
                w1 = ffmodel.dense(
                    grouped_tokens[expert_idx],
                    self.mixtral_config.intermediate_size,
                    ActiMode.AC_MODE_NONE,
                    False,
                    name=f"layers_{i}_block_sparse_moe_experts_{expert_idx}_w1",
                )
                w3 = ffmodel.dense(
                    grouped_tokens[expert_idx],
                    self.mixtral_config.intermediate_size,
                    ActiMode.AC_MODE_NONE,
                    False,
                    name=f"layers_{i}_block_sparse_moe_experts_{expert_idx}_w3",
                )
                multi = ffmodel.sigmoid_silu_multi(
                    w1,
                    w3,
                    name=f"layers_{i}_block_sparse_moe_experts_{expert_idx}_ssm",
                )
                w2 = ffmodel.dense(
                    multi,
                    self.mixtral_config.hidden_size,
                    ActiMode.AC_MODE_NONE,
                    False,
                    name=f"layers_{i}_block_sparse_moe_experts_{expert_idx}_w2",
                )
                expert_predictions.append(w2)
            assert len(expert_predictions) == self.mixtral_config.num_local_experts
            topk_values_reduced = ffmodel.reduce_sum(topk_values, [0], keepdims=True)
            topk_values = ffmodel.divide(topk_values, topk_values_reduced)
            mlp_out = ffmodel.aggregate(
                topk_values,
                topk_indices,
                expert_predictions,
                self.mixtral_config.num_local_experts,
                name=f"layers_{i}_block_sparse_moe_experts_aggregate",
            )

        _, token = ffmodel.residual_rms_norm(
            token,
            mlp_out,
            self.mixtral_config.rms_norm_eps,
            self.mixtral_config.hidden_size,
            name="norm",
        )
        dense = ffmodel.dense(
            token,
            self.mixtral_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(dense, -1)
            # output = ffmodel.beam_top_k(softmax, self.mixtral_config.max_beam_width, False)
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
            name = name.replace(".", "_").replace("model_", "")
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
