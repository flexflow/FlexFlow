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
import re


class FalconConfig:
    def __init__(self, hf_config):
        # self.max_seq_len = 256
        # self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.max_spec_tree_token_num = 64
        self.bias = hf_config.bias
        self.hidden_size = hf_config.hidden_size
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.multi_query = hf_config.multi_query
        self.n_head = (
            hf_config.n_head
            if "n_head" in hf_config.__dict__
            else hf_config.num_attention_heads
        )
        self.n_head_kv = hf_config.n_head_kv if "n_head_kv" in hf_config.__dict__ else 1
        self.n_layer = (
            hf_config.n_layer
            if "n_layer" in hf_config.__dict__
            else hf_config.num_hidden_layers
        )
        self.parallel_attn = hf_config.parallel_attn
        self.vocab_size = hf_config.vocab_size
        # Standardized FlexFlow num heads fields below
        self.num_attention_heads = self.n_head
        self.num_key_value_heads = self.n_head_kv


class FlexFlowFalcon(FlexFlowModel):
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
        self.falcon_config = FalconConfig(hf_config)
        # self.falcon_config.max_seq_length = max_seq_length
        # self.falcon_config.max_num_tokens = max_tokens_per_batch
        self.weights_filepath = weights_filepath
        self.tokenizer_filepath = tokenizer_filepath
        self.maxint = 2**31 - 1
        max_verify_tokens_per_batch = max_tokens_per_batch + self.falcon_config.max_spec_tree_token_num

        # Sanity checks
        if self.falcon_config.hidden_size % self.falcon_config.n_head != 0:
            raise ValueError(
                f"Hidden size ({self.falcon_config.hidden_size}) is not divisible by n_head ({self.falcon_config.n_head})"
            )
        if (
            self.falcon_config.n_head < self.ffconfig.tensor_parallelism_degree
            or self.falcon_config.n_head % self.ffconfig.tensor_parallelism_degree != 0
        ):
            raise ValueError(
                f"Number of q attention heads ({self.falcon_config.n_head}) is smaller, or not divisible by tensor parallelism degree ({self.ffconfig.tensor_parallelism_degree})"
            )

        self.build_model(max_tokens_per_batch if self.mode == InferenceMode.INC_DECODING_MODE else max_verify_tokens_per_batch)

    def build_model(self, max_tokens_per_batch):
        ffmodel = FFModel(self.ffconfig)

        tokens_dims = [max_tokens_per_batch, 1]
        input_tensor = ffmodel.create_tensor(tokens_dims, DataType.DT_INT32)

        embed_init = UniformInitializer(random.randint(0, self.maxint), 0, 0)
        token = ffmodel.embedding(
            input_tensor,
            self.falcon_config.vocab_size,
            self.falcon_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="word_embeddings",
        )
        axes = [
            0,
        ]

        for i in range(self.falcon_config.n_layer):
            ffmodel.set_transformer_layer_id(i)

            if i == 0:
                att_norm = ffmodel.layer_norm(
                    token,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers.{i}.input_layernorm",
                )
            else:
                token, att_norm = ffmodel.residual_layer_norm(
                    token,
                    mha,
                    mlp_output,
                    True,
                    axes,
                    True,
                    self.falcon_config.layer_norm_epsilon,
                    name=f"layers.{i}.input_layernorm",
                )

            if self.mode == InferenceMode.BEAM_SEARCH_MODE:
                mha = ffmodel.spec_inc_multiquery_self_attention(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    self.falcon_config.n_head_kv,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers.{i}.self_attention",
                )
            elif self.mode == InferenceMode.TREE_VERIFY_MODE:
                mha = ffmodel.inc_multiquery_self_attention_verify(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    self.falcon_config.n_head_kv,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers.{i}.self_attention",
                )
            elif self.mode == InferenceMode.INC_DECODING_MODE:
                mha = ffmodel.inc_multiquery_self_attention(
                    att_norm,
                    self.falcon_config.hidden_size,
                    self.falcon_config.n_head,
                    self.falcon_config.n_head_kv,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    self.falcon_config.hidden_size // self.falcon_config.n_head,
                    0.0,  # dropout
                    False,  # qkv_bias
                    False,  # final_bias
                    False,  # add_zero_attn
                    DataType.DT_NONE,  # data_type
                    None,  # kernel initializer
                    True,  # apply_rotary_embedding
                    name=f"layers.{i}.self_attention",
                )
            else:
                assert False

            dense_h_to_4h = ffmodel.dense(
                att_norm,
                self.falcon_config.hidden_size * 4,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.dense_h_to_4h",
            )
            dense_h_to_4h = ffmodel.gelu(dense_h_to_4h)
            mlp_output = ffmodel.dense(
                dense_h_to_4h,
                self.falcon_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                False,
                name=f"layers.{i}.mlp.dense_4h_to_h",
            )

        _, ln_f = ffmodel.residual_layer_norm(
            token,
            mha,
            mlp_output,
            True,
            axes,
            True,
            self.falcon_config.layer_norm_epsilon,
            name="ln_f",
        )
        lm_head = ffmodel.dense(
            ln_f,
            self.falcon_config.vocab_size,
            ActiMode.AC_MODE_NONE,
            False,
            name="lm_head",
        )

        if self.mode == InferenceMode.BEAM_SEARCH_MODE:
            softmax = ffmodel.softmax(lm_head, -1)
            # output = ffmodel.beam_top_k(softmax, self.falcon_config.max_beam_width, False)
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

    # TODO: finish this
    def convert_hf_weight_name(name):
        return (name.replace("transformer.h.", "layers.")
            .replace("transformer.", "")
            .replace("self_attention.dense", "self_attention.o_proj")
        )

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        n_head = (
            model.config.n_head
            if "n_head" in model.config.__dict__
            else model.config.num_attention_heads
        )
        for name, params in model.named_parameters():
            name = FlexFlowFalcon.convert_hf_weight_name(name)
            # Split Q,K,V attention weights
            if "self_attention.query_key_value" in name:
                name_q = name.replace("self_attention.query_key_value", "self_attention.q_proj")
                name_k = name.replace("self_attention.query_key_value", "self_attention.k_proj")
                name_v = name.replace("self_attention.query_key_value", "self_attention.v_proj")
                q, k, v = torch.split(
                    params,
                    [
                        model.config.hidden_size,
                        model.config.hidden_size // n_head,
                        model.config.hidden_size // n_head,
                    ],
                    0,
                )
                q.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_q))
                k.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_k))
                v.detach().cpu().numpy().tofile(os.path.join(dst_folder, name_v))
            else:
                params.detach().cpu().numpy().tofile(os.path.join(dst_folder, name))
        # LM head weight
        model.lm_head.weight.detach().cpu().numpy().tofile(
            os.path.join(dst_folder, "lm_head.weight")
        )

    
    def convert_ff_weight_name(name):
        
        converted_name = name
        converted_name = converted_name.replace("mlp_dense_h_to_4h", "mlp.dense_h_to_4h")
        converted_name = converted_name.replace("mlp_dense_4h_to_h", "mlp.dense_4h_to_h")
        converted_name = converted_name.replace("attention_wo", "self_attention.dense")
        if name.startswith("ln") or name.startswith("word_embeddings"):
            converted_name = "transformer." + converted_name
        converted_name = re.sub(r"layers_(\d+)_", r"transformer.h.\1.", converted_name)
        converted_name = re.sub(r"_(bias|weight)$", r".\1", converted_name)

        return converted_name


    def load_weights_into_hf_model(model, src_folder):
        """
        Load weights from a specified folder and apply them to a Hugging Face model.
        
        Parameters:
        - model: The instance of the Hugging Face model to load the weights into.
        - src_folder: The path to the folder containing the weight files.
        - config: The configuration object for the model.
        """
        
        hidden_size = model.config.hidden_size
        n_head = (
            model.config.n_head
            if "n_head" in model.config.__dict__
            else model.config.num_attention_heads
        )
        
        print("Model hidden size:", hidden_size)
        print("Model num_attention_heads:", n_head)
        
        num_attention_heads = n_head
        hidden_size_per_head = hidden_size // n_head
        intermediate_size = hidden_size * 4
        
        qkv_weights = {}
        
        for file_name in os.listdir(src_folder):
            weight_path = os.path.join(src_folder, file_name)
            print("\nProcessing weight file:", weight_path)
            original_name = FlexFlowFalcon.convert_ff_weight_name(file_name.replace('.bin', ''))
            print("Converted weight name:", original_name)
            
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weight file found for {file_name}")
            
            weight_data = np.fromfile(weight_path, dtype=np.float16).astype(np.float32)
            print(f"Data type after conversion: {weight_data.dtype}, Size: {weight_data.size}")
            
            # for q,k,v weights, store in dict
            if ("attention_wq" in original_name) or ("attention_wk" in original_name) or ("attention_wv" in original_name):
                qkv_match = re.search("(wq|wk|wv)", file_name)
                qkv_type = qkv_match.group(0) if qkv_match else None
                layer_num_match = re.search(r"transformer.h.(\d+)", original_name)
                layer_num = int(layer_num_match.group(1)) if layer_num_match else None
                print(f"QKV type: {qkv_type}, Layer number: {layer_num}")
                
                if layer_num is not None:
                    if layer_num not in qkv_weights:
                        # qkv_shape = (hidden_size_per_head * num_attention_heads, hidden_size)
                        per_type_space = hidden_size_per_head * n_head
                        
                        qkv_name = f"transformer.h.{layer_num}.self_attention.query_key_value.weight"
                        if qkv_name in model.state_dict():
                            qkv_param_size = model.state_dict()[qkv_name].shape[0]
                        qkv_shape = (qkv_param_size, hidden_size)
                        qkv_weights[layer_num] = np.zeros(qkv_shape)
                        print(f"Initialized QKV shape for layer {layer_num}: {qkv_shape}")
                        
                    type_index = {"wq": 0, "wk": 1, "wv": 2}.get(qkv_type, 0)
                    offset = type_index * per_type_space
                    print("offset for this weight is: ", offset)
                    ## dim 0 sizes: 
                    dim_wq = hidden_size
                    dim_wk = hidden_size // n_head
                    dim_wv = hidden_size // n_head
                    print(dim_wq, dim_wk, dim_wv)
                    
                    try:
                        expected_shape = (weight_data.size // hidden_size, hidden_size)
                        reshaped_data = weight_data.reshape(expected_shape)
                        print(f"Reshaped QKV weights for {qkv_type} in layer {layer_num} with shape {expected_shape}.")
                    except ValueError as e:
                        print(f"Error reshaping {qkv_type} weights for layer {layer_num}: {e}")
                        print(f"Attempting to reshape data of size {weight_data.size} into shape (-1, {hidden_size})")
                        
                        
                    try:
                        if qkv_type == "wq":
                            qkv_weights[layer_num][0:dim_wq, :] = reshaped_data
                        elif qkv_type == "wk":
                            qkv_weights[layer_num][dim_wq:dim_wk+dim_wq, :] = reshaped_data
                        else:
                            qkv_weights[layer_num][dim_wq+dim_wk:, :] = reshaped_data
                    except ValueError as e:
                        print(f"Error assigning {qkv_type} weights for layer {layer_num}: {e}")
                continue
            
            # for weights that are not q,k,v, get the param names
            param = model.state_dict().get(original_name, None)
            if param is None:
                print(f"Warning: {original_name} not found i