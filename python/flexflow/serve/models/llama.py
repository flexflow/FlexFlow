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
        self.max_seq_len = 256
        self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = hf_config.num_attention_heads if hf_config.num_key_value_heads is None else hf_config.num_key_value_heads
        self.hidden_size = hf_config.hidden_size
        self.rms_norm_eps = hf_config.rms_norm_eps
        self.intermediate_size = hf_config.intermediate_size


class FlexFlowLLAMA(FlexFlowModel):
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
        self.llama_config = LLAMAConfig(hf_config)
        self.llama_config.max_seq_length = max_seq_length
        self.llama_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1

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

        self.build_model()

    def build_model(self):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [self.llama_config.max_num_tokens, 1]
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
            name="tok_embeddings_weight",
        )

        for i in range(self.llama_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            attn_norm = ffmodel.rms_norm(
                token,
                self.llama_config.rms_norm_eps,
                self.llama_config.hidden_size,
                name=f"layers_{i}_attention_norm_weight",
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
                    False,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention_weight",
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
                    False,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention_weight",
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
                    False,  # bias
                    False,  # add_bias_kv
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers_{i}_attention_weight",
                )
            else:
                assert False

            token = ffmodel.add(token, mha)
            ff_norm = ffmodel.rms_norm(
                token,
                self.llama_config.rms_norm_eps,
                self.llama_config.hidden_size,
                name=f"layers_{i}_ffn_norm_weight",
            )
            w1 = ffmodel.dense(
                ff_norm,
                self.llama_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w1_weight",
            )
            w3 = ffmodel.dense(
                ff_norm,
                self.llama_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w3_weight",
            )
            sigmoid = ffmodel.sigmoid(w1)
            silu = ffmodel.multiply(w1, sigmoid)
            multi = ffmodel.multiply(silu, w3)
            w2 = ffmodel.dense(
                multi,
                self.llama_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers_{i}_feed_forward_w2_weight",
            )
            token = ffmodel.add(token, w2)

        token = ffmodel.rms_norm(
            token,
            self.llama_config.rms_norm_eps,
            self.llama_config.hidden_size,
            name="norm_weight",
        )
        dense = ffmodel.dense(
            token,
            self.llama_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="output_weight",
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

    def get_layers_with_weights(self):
        layer_names = ["tok_embeddings_weight", "norm_weight", "output_weight"] + [
            expr
            for i in range(self.llama_config.num_hidden_layers)
            for expr in (
                f"layers_{i}_attention_norm_weight",
                f"layers_{i}_attention_weight",
                f"layers_{i}_ffn_norm_weight",
                f"layers_{i}_feed_forward_w1_weight",
                f"layers_{i}_feed_forward_w3_weight",
                f"layers_{i}_feed_forward_w2_weight",
            )
        ]
        layers_with_weights = {
            layer_name: self.ffmodel.get_layer_by_name(layer_name)
            for layer_name in layer_names
        }

        return layers_with_weights
