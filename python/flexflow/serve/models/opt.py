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
import re
import torch


class OPTConfig:
    def __init__(self, hf_config):
        # self.max_seq_len = 256
        # self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 64
        self.do_layer_norm_before = hf_config.do_layer_norm_before
        self.dropout = hf_config.dropout
        self.enable_bias = hf_config.enable_bias
        self.ffn_dim = hf_config.ffn_dim
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_elementwise_affine = hf_config.layer_norm_elementwise_affine
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.vocab_size = hf_config.vocab_size
        self.word_embed_proj_dim = hf_config.word_embed_proj_dim
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = hf_config.num_attention_heads


class FlexFlowOPT(FlexFlowModel):
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
        self.opt_config = OPTConfig(hf_config)
        # self.opt_config.max_seq_length = max_seq_length
        # self.opt_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = max_tokens_per_batch + self.opt_config.max_spec_tree_token_num

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

        self.build_model(max_tokens_per_batch if self.mode == InferenceMode.INC_DECODING_MODE else max_verify_tokens_per_batch)

    def build_model(self, max_tokens_per_batch):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [max_tokens_per_batch, 1]
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
            name="embed_tokens",
        )
        positional_embedding = ffmodel.embedding(
            position_tensor,
            self.opt_config.max_position_embeddings,
            self.opt_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="embed_positions",
        )

        axes = [
            0,
        ]

        for i in range(self.opt_config.num_hidden_layers):
            ffmodel.set_transformer_layer_id(i)

            if self.opt_config.do_layer_norm_before:
                residual, hidden_states = ffmodel.residual_layer_norm(
                    token if i == 0 else residual,
                    positional_embedding if i == 0 else fc2,
                    None,
                    False,
                    axes,
                    self.opt_config.layer_norm_elementwise_affine,
                    1e-05,
                    name=f"layers.{i}.self_attn_layer_norm",
                )
            else:
                hidden_states = ffmodel.add(token, positional_embedding)
                residual = hidden_states

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multihead_self_attention(
                    hidden_states,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    0.0,  # dropout
                    True,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.opt_config.hidden_size / self.opt_config.num_attention_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers.{i}.self_attn",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multihead_self_attention_verify(
                    hidden_states,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    0.0,  # dropout
                    True,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.opt_config.hidden_size / self.opt_config.num_attention_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers.{i}.self_attn",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multihead_self_attention(
                    hidden_states,
                    self.opt_config.hidden_size,
                    self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    self.opt_config.hidden_size // self.opt_config.num_attention_heads,
                    0.0,  # dropout
                    True,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    False,  # apply_rotary_embedding
                    True,  # scaling_query
                    (self.opt_config.hidden_size / self.opt_config.num_attention_heads)
                    ** (-0.5),  # scaling_factor
                    False,  # qk_prod_scaling
                    name=f"layers.{i}.self_attn",
                )
            else:
                assert False

            # This is either a before or after attention LayerNorm. In both cases, we need to compute the LN here.
            residual, ff_norm = ffmodel.add_bias_residual_layer_norm(
                mha,
                residual,
                axes,
                self.opt_config.layer_norm_elementwise_affine,
                1e-05,
                name=f"layers.{i}.add_bias_residual_layer_norm",
            )

            if not self.opt_config.do_layer_norm_before:
                residual = ff_norm

            fc1 = ffmodel.dense(
                ff_norm,
                self.opt_config.ffn_dim,
                ActiMode.AC_MODE_RELU,
                True,
                name=f"layers.{i}.fc1",
            )
            fc2 = ffmodel.dense(
                fc1,
                self.opt_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers.{i}.fc2",
            )

            if not self.opt_config.do_layer_norm_before:
                _, residual = ffmodel.residual_layer_norm(
                    residual,
                    fc2,
                    None,
                    False,
                    axes,
                    self.opt_config.layer_norm_elementwise_affine,
                    1e-05,
                    name=f"layers.{i}.final_layer_norm",
                )

        _, all_final_norm = ffmodel.residual_layer_norm(
            residual,
            fc2,
            None,
            False,
            axes,
            self.opt_config.layer_norm_elementwise_affine,
            1e-05,
            name=f"final_layer_norm",
        )
        lm_head = ffmodel.dense(
            all_final_norm,
            self.opt_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
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
                softmax = ffmodel.softmax(lm_head, -1)
                output = ffmodel.argmax(softmax, False)

        self.ffmodel = ffmodel

    def convert_hf_weight_name(name):
        return (
            name.replace("decoder.", "")
            .replace("model.", "")
            .replace("self_attn.out_proj", "self_attn.o_proj")
            .replace("self_attn.o_proj.bias", "add_bias_residual_layer_norm.attn_bias")
            .replace(
                ".final_layer_norm", ".add_bias_residual_layer_norm"
            )  # important to use the leading "_" to avoid matching the last LayerNorm
        )

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for name, params in model.named_parameters():
            name = FlexFlowOPT.convert_hf_weight_name(name)
            params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")
        # copy embedding weights
        shutil.copy(
            os.path.join(dst_folder, "embed_tokens.weight"),
            os.path.join(dst_folder, "lm_head.weight"),
        )
        
    def convert_ff_weight_name(name):
        # Reverse the previous conversion rules
        converted_name = (
            name
            .replace("add_bias_residual_layer_norm.attn_bias", "attention_wo_bias")
            .replace(".add_bias_residual_layer_norm", ".final_layer_norm")
            .replace("wq", "q_proj")
            .replace("wk", "k_proj")
            .replace("wv", "v_proj")
            .replace("wo", "out_proj")
            .replace("self_attn.o_proj", "self_attn.out_proj")
            .replace("attention", "self_attn")
        )
        
        converted_name = re.sub(r"layers_(\d+)_", r"layers.\1.", converted_name)
        converted_name = re.sub(r"_(bias|weight)$", r".\1", converted_name)
        converted_name = re.sub(r"self_attn_(?!layer_norm)", "self_attn.", converted_name)
        converted_name = converted_name.replace("embed_tokens_weight_lm_head", "embed_tokens.weight")
        
        # Prepend "model.decoder." to the weight name
        if not converted_name.startswith("model.decoder.") and "lm_head" not in converted_name:
            converted_name = "model.decoder." + converted_name
        
        return converted_name


    def load_weights_into_hf_model(model, src_folder):
        """
        Load weights from a specified folder and apply them to a Hugging Face model.
        
        This function iterates through the weight files in the specified folder, 
        converts the FlexFlow weight names to Hugging Face format, and loads the 
        weights into the Hugging Face model. It handles special cases like shape 
        mismatches by adjusting the weights accordingly.
        
        Parameters:
        - model: The instance of the Hugging Face model to load the weights into.
        - src_folder: The path to the folder containing the weight files.
        """
        
        for file_name in os.listdir(src_folder):
            weight_path = os.path.join(src_folder, file_name)
            print("Converting weight name:", weight_path)
            
            if weight_path.endswith("rev_sha.txt"):
                print("Skipping rev_sha.txt")
                continue

            original_name = FlexFlowOPT.convert_ff_weight_name(file_name.replace('.bin', ''))
            print("Original name of the weights is:", original_name)
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weight file found for {file_name}")
            
            weight_data = np.fromfile(weight_path, dtype=np.float16).astype(np.float32)
            if original_name not in model.state_dict():
                raise KeyError(f"Parameter {original_name} not found in model.")
            param = model.state_dict()[original_name]
            
            # Calculate the reshape size automatically based on expected parameter size
            expected_numel = param.numel()
            if weight_data.size != expected_numel:
                print(f"Adjusting shape for {original_name} from {weight_data.size} to {expected_numel}")
                # Check if weight_data can be evenly divided by expected_numel
                if weight_data.size % expected_numel == 0:
                    # Determine the reshape size
                    factor = weight_data.size // expected_numel
                    new_shape = (factor, ) + tuple(param.shape)
                    weight_data_reshaped = weight_data.reshape(new_shape)
                    weight_tensor = torch.from_numpy(weight_data_reshaped[0])
                else:
                    raise ValueError(f"Cannot adjust shape for {original_name} due to incompatible size.")
            else:
                weight_tensor = torch.from_numpy(weight_data).reshape(param.shape)
            
            with torch.no_grad():
                model.state_dict()[original_name].copy_(weight_tensor)