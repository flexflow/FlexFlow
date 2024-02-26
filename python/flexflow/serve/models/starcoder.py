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
        # self.max_seq_len = 256
        # self.max_num_tokens = 64
        self.max_beam_width = 1
        self.max_beam_depth = 8
        self.dropout_p = hf_config.attn_pdrop
        self.hidden_size = hf_config.n_embd
        self.layer_norm_epsilon = hf_config.layer_norm_epsilon
        self.max_position_embeddings = hf_config.n_positions
        self.num_hidden_layers = hf_config.n_layer
        self.vocab_size = hf_config.vocab_size
        self.intermediate_size = hf_config.n_inner
        self.n_head_kv = 1 if hf_config.multi_query else hf_config.n_head
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
        self.starcoder_config = STARCODERConfig(hf_config)
        # self.starcoder_config.max_seq_length = max_seq_length
        # self.starcoder_config.max_num_tokens = max_tokens_per_batch
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

        self.build_model(max_tokens_per_batch)

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
            name="transformer_wte",
        )
        positional_embedding = ffmodel.embedding(
            position_tensor,
            self.starcoder_config.max_position_embeddings,
            self.starcoder_config.hidden_size,
            AggrMode.AGGR_MODE_NONE,
            self.data_type,
            None,
            embed_init,
            name="transformer_wpe",
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
                name=f"layers_{i}_ln_1",
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
                True,  # qkv_bias
                False,  # final_bias
                False,  # add_zero_attn
                DataType.DT_NONE,  # data_type
                None,  # kernel initializer
                False,  # apply_rotary_embedding
                name=f"layers_{i}_attention",
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
                name=f"layers_{i}_ln_2",
            )

            # mlp

            c_fc = ffmodel.dense(
                l2_norm,
                self.starcoder_config.intermediate_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers_{i}_mlp_c_fc",
            )
            activation = ffmodel.gelu(c_fc, False)
            c_proj = ffmodel.dense(
                activation,
                self.starcoder_config.hidden_size,
                ActiMode.AC_MODE_NONE,
                True,
                name=f"layers_{i}_mlp_c_proj",
            )

        _, ln_f = ffmodel.residual_layer_norm(
            residual,
            c_proj,
            None,
            False,
            axes,
            True,
            self.starcoder_config.layer_norm_epsilon,
            name=f"transformer_ln_f",
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
        
        
    def convert_ff_weight_name(name):
        """
        Convert weight names from FlexFlow format back to Hugging Face format.
        """
        # Example conversion logic, adjust as needed
        if "attention_wq" in name or "attention_wk" in name or "attention_wv" in name:
            converted_name = converted_name.replace("attention_wq", "attn.c_attn").replace("attention_wk", "attn.c_attn").replace("attention_wv", "attn.c_attn")
        elif "attention_wo" in name:
            converted_name = converted_name.replace("attention_wo", "attn.c_proj")
        
        converted_name = re.sub(r"layers_(\d+)_", r"transformer.h.\1.", converted_name)

        return converted_name
    
    
    def load_weights_into_hf_model(model, src_folder):
        """
        Load weights from a specified folder and apply them to a Hugging Face model.

        Parameters:
        - model: The instance of the Hugging Face model to load the weights into.
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

            weight_data = np.fromfile(weight_path, dtype=np.float32)

            # Find the parameter in the model
            param = model.state_dict().get(original_name)
            if param is None:
                print(f"Warning: {original_name} not found in model parameters.")
                continue

            # Special handling for q, k, v weights
            if ("attention_wq" in original_name) or ("attention_wk" in original_name) or ("attention_wv" in original_name):
                qkv_match = re.search("(wq|wk|wv)", file_name)
                qkv_type = qkv_match.group(0) if qkv_match else None
                layer_num_match = re.search(r"transformer.h.(\d+)", original_name)
                layer_num = int(layer_num_match.group(1)) if layer_num_match else None
                print(f"QKV type: {qkv_type}, Layer number: {layer_num}")
                
                if layer_num is not None:
                    if layer_num not in qkv_weights:
                        
                        qkv_name = f"transformer.h.{layer_num}.self_attention.query_key_value.weight"
                        if qkv_name in model.state_dict():
                            qkv_param_size = model.state_dict()[qkv_name].shape[0]
                        qkv_shape = (qkv_param_size, hidden_size)
                        qkv_weights[layer_num] = np.zeros(qkv_shape)
                        print(f"Initialized QKV shape for layer {layer_num}: {qkv_shape}")
                        
                    type_index = {"wq": 0, "wk": 1, "wv": 2}.get(qkv_type, 0)
                    ## dim 0 sizes: 
                    dim_wq = hidden_size
                    dim_wk = hidden_size // n_head
                    dim_wv = hidden_size // n_head
                    
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


            # Handle other parameters
            param = model.state_dict().get(original_name)
            if param is None:
                print(f"Warning: {original_name} not found in model parameters.")
                continue
            reshaped_weight_data = weight_data.reshape(param.shape)
            param.data.copy_(torch.from_numpy(reshaped_weight_data))
            
        
        # Assign the combined QKV weights to the model
        for layer_num, weight in qkv_weights.items():
            qkv_name = f"transformer.h.{layer_num}.self_attention.query_key_value.weight"
            if qkv_name in model.state_dict():
                param = model.state_dict()[qkv_name]
                # Ensure the combined weight is correctly reshaped to fit the model's expectations
                param.data.copy_(torch.from_numpy(weight.reshape(param.shape)))
                
       
