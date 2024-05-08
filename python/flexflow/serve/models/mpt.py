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
import random, torch, shutil, os, re
import numpy as np


class MPTConfig:
    def __init__(self, hf_config):
        # self.max_seq_len = 256
        # self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 20
        self.hidden_size = hf_config.d_model
        self.n_heads = hf_config.n_heads
        self.n_layers = hf_config.n_layers
        self.vocab_size = hf_config.vocab_size
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.n_heads
        self.num_key_value_heads = hf_config.n_heads


class FlexFlowMPT(FlexFlowModel):
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
        self.mpt_config = MPTConfig(hf_config)
        # self.mpt_config.max_seq_length = max_seq_length
        # self.mpt_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = (
            max_tokens_per_batch + self.mpt_config.max_spec_tree_token_num
        )

        # Sanity checks
        if self.mpt_config.hidden_size % self.mpt_config.n_heads != 0:
            raise ValueError(
                f"Hidden size ({self.mpt_config.hidden_size}) is not divisible by n_head ({self.mpt_config.n_heads})"
            )

        # Sanity checks
        if (
            self.mpt_config.n_heads < self.ffconfig.tensor_parallelism_degree
            or self.mpt_config.n_heads % self.ffconfig.tensor_parallelism_degree != 0
        ):
            raise ValueError(
                f"Number of attention heads ({self.mpt_config.n_heads}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )
        self.build_model(
            max_tokens_per_batch
            if self.mode == InferenceMode.INC_DECODING_MODE
            else max_verify_tokens_per_batch
        )

    def build_model(self, max_tokens_per_batch):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [max_tokens_per_batch, 1]
        input = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        hidden_states = ffmodel.embedding(
            input,
            self.mpt_config.vocab_size,
            self.mpt_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="wte",
        )

        axes = [
            0,
        ]

        for i in range(self.mpt_config.n_layers):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                layernorm_output = ffmodel.layer_norm(
                    hidden_states,
                    axes,
                    True,
                    1e-05,
                    False,
                    name=f"layers.{i}.norm_1",
                )
            else:
                hidden_states, layernorm_output = ffmodel.residual_layer_norm(
                    intermediate_output,
                    hidden_states,
                    None,
                    False,
                    axes,
                    True,
                    1e-05,
                    False,
                    name=f"layers.{i}.norm_1",
                )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                attn_outputs = ffmodel.spec_inc_multihead_self_attention(
                    layernorm_output,
                    self.mpt_config.hidden_size,
                    self.mpt_config.n_heads,
                    self.mpt_config.hidden_size // self.mpt_config.n_heads,
                    self.mpt_config.hidden_size // self.mpt_config.n_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.mpt_config.hidden_size / self.mpt_config.n_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    True,  # qk_prod_scaling
                    name=f"layers.{i}.attn",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                attn_outputs = ffmodel.inc_multihead_self_attention_verify(
                    layernorm_output,
                    self.mpt_config.hidden_size,
                    self.mpt_config.n_heads,
                    self.mpt_config.hidden_size // self.mpt_config.n_heads,
                    self.mpt_config.hidden_size // self.mpt_config.n_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.mpt_config.hidden_size / self.mpt_config.n_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    True,  # qk_prod_scaling
                    name=f"layers.{i}.attn",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                attn_outputs = ffmodel.inc_multihead_self_attention(
                    layernorm_output,
                    self.mpt_config.hidden_size,
                    self.mpt_config.n_heads,
                    self.mpt_config.hidden_size // self.mpt_config.n_heads,
                    self.mpt_config.hidden_size // self.mpt_config.n_heads,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.mpt_config.hidden_size / self.mpt_config.n_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    True,  # qk_prod_scaling
                    name=f"layers.{i}.attn",
                )
            else:
                assert False

            hidden_states, layernorm_output = ffmodel.residual_layer_norm(
                attn_outputs,
                hidden_states,
                None,
                False,
                axes,
                True,
                1e-05,
                False,
                name=f"layers.{i}.norm_2",
            )
            # mlp
            layernorm_output = ffmodel.dense(
                layernorm_output,
                4 * self.mpt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.ffn.up_proj",
            )
            layernorm_output = ffmodel.gelu(layernorm_output)
            intermediate_output = ffmodel.dense(
                layernorm_output,
                self.mpt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.ffn.down_proj",
            )

        _, all_final_norm = ffmodel.residual_layer_norm(
            intermediate_output,
            hidden_states,
            None,
            False,
            axes,
            True,
            1e-05,
            False,
            name=f"norm_f",
        )
        lm_head = ffmodel.dense(
            all_final_norm,
            self.mpt_config.vocab_size,
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

    def convert_hf_weight_name(name):
        return (
            name.replace("transformer.blocks.", "layers.")
            .replace("transformer.", "")
            .replace("attn.out_proj", "attn.o_proj")
        )

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = FlexFlowMPT.convert_hf_weight_name(name)
            if "Wqkv" in name:
                name_q = name.replace("attn.Wqkv", "attn.q_proj")
                name_k = name.replace("attn.Wqkv", "attn.k_proj")
                name_v = name.replace("attn.Wqkv", "attn.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.d_model,
                        model.config.d_model,
                        model.config.d_model,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))

        shutil.copy(
            os.path.join(dst_folder, "wte.weight"),
            os.path.join(dst_folder, "lm_head.weight"),
        )


    def convert_ff_weight_name(name):
        # Reverses the conversion logic for MPT model weights
        converted_name = name
        if "norm_f" in converted_name or "wte" in converted_name:
            converted_name = converted_name.replace("_", ".").replace("norm.f", "norm_f")
            
        converted_name = converted_name.replace("attn.o_proj", "attn.out_proj")
        converted_name = converted_name.replace("ffn_", "ffn.")
        converted_name = re.sub(r"layers.(\d+).", r"transformer.blocks.\1.", converted_name)
        converted_name = re.sub(r"_(bias|weight)$", r".\1", converted_name)
        
        if ("wte" in converted_name) or ("norm_f" in converted_name):
            converted_name = "transformer." + converted_name
        
        return converted_name

    def load_weights_into_hf_model(model, src_folder):
        """
        Load weights from a specified folder and apply them to a Hugging Face MPT model.
        
        Parameters:
        - model: The instance of the Hugging Face model to load the weights into.
        - src_folder: The path to the folder containing the weight files.
        """
        
        d_model = model.config.d_model
        print("dimension of the model is: ", d_model)
        
        qkv_weights = {}
        
        for file_name in os.listdir(src_folder):
            weight_path = os.path.join(src_folder, file_name)
            if weight_path.endswith("rev_sha.txt"):
                print("skipping rev_sha.txt")
                continue
            elif "lm_head" in weight_path:
                print("skipping lm_head.weight")
                continue
            else:
                original_name = FlexFlowMPT.convert_ff_weight_name(file_name.replace('.bin', ''))
                print("\nconverting weights name of: ", file_name, "to ", original_name)
                
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weight file found for {file_name}")
            
            weight_data = np.fromfile(weight_path, dtype=np.float32)
            print(f"Data type after conversion: {weight_data.dtype}, Size: {weight_data.size}")
            
            # Special handling for combined QKV weights
            if ("q_proj" in file_name) or ("k_proj" in file_name) or ("v_proj" in file_name):
                layer_num_match = re.search(r"layers\.(\d+)", original_name)
                layer_num = int(layer_num_match.group(1)) if layer_num_match else None
                qkv_type = original_name.split("_")[-2]
                
                if layer_num is not None:
                    qkv_key = f"layers.{layer_num}.attn_Wqkv"
                    # initialize qkv layer in dict
                    if qkv_key not in qkv_weights:
                        qkv_weights[qkv_key] = {'wq': None, 'wk': None, 'wv': None}
                        print(f"Initialized QKV layer {layer_num}")
                    # assign weights into dict
                    qkv_weights[qkv_key][qkv_type] = weight_data
                
                continue
            
            # for weights that are not q,k,v, get the param names and assign weights accordingly
            param = model.state_dict().get(original_name, None)
            if weight_data.size != param.numel():
                raise ValueError(f"Shape mismatch for {original_name}, model expects {param.numel()} elements, got {weight_data.size}")
            
            weight_tensor = torch.from_numpy(weight_data).reshape(param.shape)
            with torch.no_grad():
                model.state_dict()[original_name].copy_(weight_tensor)

                    
        for qkv_key, weights_dict in qkv_weights.items():
            wq, wk, wv = weights_dict['wq'], weights_dict['wk'], weights_dict['wv']
            if None in (wq, wk, wv):
                raise ValueError(f"Missing weights for {qkv_key}")

            combined_qkv = np.concatenate([wq, wk, wv], axis=0)
            qkv_name = qkv_key.replace("layers.", "transformer.blocks.")+".weight"
            
            param_shape = model.state_dict()[qkv_name].shape
            combined_qkv_reshaped = combined_qkv.reshape(param_shape)

            model.state_dict()[qkv_name].copy_(torch.from_numpy(combined_qkv_reshaped))

            print(f"Assigned combined QKV weights to {qkv_key}.")
                