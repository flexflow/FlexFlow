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
import re
import os
import numpy as np
import torch


class LLAMAConfig:
    def __init__(self, hf_config):
        # self.max_seq_len = 256
        # self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 64
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
        max_verify_tokens_per_batch = max_tokens_per_batch + self.llama_config.max_spec_tree_token_num
    

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

        self.build_model(max_tokens_per_batch if self.mode == InferenceMode.INC_DECODING_MODE else max_verify_tokens_per_batch)

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
                    name=f"layers.{i}.self_attn",
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
                    name=f"layers.{i}.self_attn",
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
                    name=f"layers.{i}.self_attn",
                )
            else:
                assert False

            token, ff_norm = ffmodel.residual_rms_norm(
                token,
                mha,
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


    def convert_ff_weight_name(name):
        converted_name = (
            name
            .replace("w1", "gate_proj")
            .replace("w2", "down_proj")
            .replace("w3", "up_proj")
            .replace("wq", "q_proj")
            .replace("wk", "k_proj")
            .replace("wv", "v_proj")
            .replace("wo", "o_proj")
            .replace("feed_forward_", "mlp.")
            .replace("post_self_attn", "post_attention")
            .replace("attention_norm", "input_layernorm")
            .replace("tok_embeddings", "embed_tokens")
            .replace("output", "lm_head")
            
        )
        
        converted_name = re.sub(r"layers_(\d+)_", r"layers.\1.", converted_name)
        converted_name = re.sub(r"_(bias|weight)$", r".\1", converted_name)
        # converted_name = re.sub(r"attention_(?!norm)", "self_attn.", converted_name)
        
        converted_name = converted_name.replace("ffn_norm", "post_attention_layernorm")
            
        if "lm_head" not in converted_name:
            converted_name = "model." + converted_name   
                 
        return converted_name
    
    def load_weights_into_hf_model(model, src_folder):
        """
        Load weights from a specified folder and apply them to a Hugging Face model.

        Parameters:
        - model: The instance of the Hugging Face model to load weights into.
        - src_folder: The path to the folder containing the weight files.
        """
        for file_name in os.listdir(src_folder):
            weight_path = os.path.join(src_folder, file_name)
            if weight_path.endswith("rev_sha.txt"):
                print("skipping rev_sha.txt")
                continue
            else:
                original_name = FlexFlowLLAMA.convert_ff_weight_name(file_name.replace('.bin', ''))
                print(f"Converting weight name: {file_name} to {original_name}")
            
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weight file found for {file_name}")
            
            weight_data = np.fromfile(weight_path, dtype=np.float16).astype(np.float32)
            if original_name not in model.state_dict():
                raise KeyError(f"Parameter {original_name} not found in model.")
            
            param = model.state_dict()[original_name]
            expected_numel = param.numel()
            if weight_data.size != expected_numel:
                print(f"Adjusting shape for {original_name} from {weight_data.size} to {expected_numel}.")
                if weight_data.size % expected_numel == 0:
                    # If the weight data is an exact multiple of the expected size,
                    # it's likely that the data includes redundant dimensions.
                    # We'll reshape it by keeping only the first segment that matches the expected shape.
                    factor = weight_data.size // expected_numel
                    new_shape = (factor,) + tuple(param.shape)
                    weight_data_reshaped = weight_data.reshape(new_shape)[0]  # Keep only the first segment
                    weight_tensor = torch.from_numpy(weight_data_reshaped)
                else:
                    raise ValueError(f"Cannot adjust shape for {original_name} due to incompatible size.")
            else:
                weight_tensor = torch.from_numpy(weight_data).reshape(param.shape)
            
            with torch.no_grad():
                param.copy_(weight_tensor)