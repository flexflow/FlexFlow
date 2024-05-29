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
        self.max_spec_tree_token_num = 20
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
        max_verify_tokens_per_batch = (
            max_tokens_per_batch + self.falcon_config.max_spec_tree_token_num
        )

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

    def convert_weight_name_hf2ff(name):
        return (
            name.replace("transformer.h.", "layers.")
            .replace("transformer.", "")
            .replace("self_attention.dense", "self_attention.o_proj")
        )

    def convert_weight_name_ff2hf(name):
        return "transformer." + name.replace(
            "self_attention.o_proj", "self_attention.dense"
        ).replace("layers.", "h.")

    def convert_hf_model(model, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        n_head = (
            model.config.n_head
            if "n_head" in model.config.__dict__
            else model.config.num_attention_heads
        )
        for name, params in model.named_parameters():
            name = FlexFlowFalcon.convert_weight_name_hf2ff(name)
            # Split Q,K,V attention weights
            if "self_attention.query_key_value" in name:
                name_q = name.replace(
                    "self_attention.query_key_value", "self_attention.q_proj"
                )
                name_k = name.replace(
                    "self_attention.query_key_value", "self_attention.k_proj"
                )
                name_v = name.replace(
                    "self_attention.query_key_value", "self_attention.v_proj"
                )
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

    def load_weights_into_hf_model(model, src_folder):
        """
        Load weights from a specified folder and apply them to a Hugging Face model.

        Parameters:
        - model: The instance of the Hugging Face model to load the weights into.
        - src_folder: The path to the folder containing the weight files.
        - config: The configuration object for the model.
        """

        print(f"loading weights from {model} into {src_folder}")

        hidden_size = model.config.hidden_size
        n_head = (
            model.config.n_head
            if "n_head" in model.config.__dict__
            else model.config.num_attention_heads
        )

        print("Model hidden size:", hidden_size)
        print("Model num_attention_heads:", n_head)

        # num_attention_heads = n_head
        # hidden_size_per_head = hidden_size // n_head
        # intermediate_size = hidden_size * 4

        qkv_weights = {}

        for file_name in os.listdir(src_folder):
            weight_path = os.path.join(src_folder, file_name)
            print("\nProcessing weight file:", weight_path)
            if weight_path.endswith("rev_sha.txt"):
                print("skipping rev_sha.txt")
                continue
            else:
                original_name = FlexFlowFalcon.convert_weight_name_ff2hf(file_name)
                print(f"Converted weight name from {file_name} to {original_name}")

            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weight file found for {file_name}")

            weight_data = np.fromfile(weight_path, dtype=np.float16).astype(np.float32)
            print(
                f"Data type after conversion: {weight_data.dtype}, Size: {weight_data.size}"
            )

            # for q,k,v weights, store in dict
            if (
                ("q_proj" in original_name)
                or ("k_proj" in original_name)
                or ("v_proj" in original_name)
            ):

                layer_num_match = re.search(r"transformer.h.(\d+)", original_name)
                layer_num = int(layer_num_match.group(1)) if layer_num_match else None
                qkv_type = file_name.split(".")[-2]
                print(f"qkv type for this weight is {qkv_type}")

                if layer_num is not None:
                    qkv_key = (
                        f"transformer.h.{layer_num}.self_attention.query_key_value"
                    )
                    if qkv_key not in qkv_weights:
                        qkv_weights[qkv_key] = {
                            "q_proj": None,
                            "k_proj": None,
                            "v_proj": None,
                        }

                    qkv_weights[qkv_key][qkv_type] = weight_data
                continue

            # Handle non-QKV weights normally
            param = model.state_dict()[original_name]
            expected_numel = param.numel()
            print(f"expected param shape is {expected_numel}")
            if param is None:
                # raise ValueError(f"Warning: {original_name} not found!")
                print(f"Warning: {original_name} not found!")
                continue

            if weight_data.size != param.numel():
                # print(f"shape mismatch for {original_name}, model expects {param.numel()} elements, got {weight_data.size}")
                expected_shape = param.shape
                if weight_data.size % param.numel() == 0:
                    factor = weight_data.size // np.prod(expected_shape)
                    new_shape = (factor,) + tuple(expected_shape)
                    weight_data_reshaped = weight_data.reshape(new_shape)[0]
                    weight_tensor = torch.from_numpy(weight_data_reshaped)
                else:
                    raise ValueError(
                        f"Shape mismatch and cannot convert for {original_name}"
                    )
            else:
                weight_tensor = torch.from_numpy(weight_data).reshape(param.shape)

            print(f"shape of the weight tensor is: {weight_tensor.shape}")
            with torch.no_grad():
                model.state_dict()[original_name].copy_(weight_tensor)
                print(f"Assigned weight {original_name} successfully!\n")

        # Assign combined QKV weights
        for qkv_name, weights_dict in qkv_weights.items():
            print("\n========= Processing combined QKV weights ==========")
            print(
                f"qkv name is {qkv_name}, hidden size is {hidden_size}, number of attention heads is {n_head}"
            )
            print(
                f"the weights dimensions are: {weights_dict['q_proj'].shape}, {weights_dict['k_proj'].shape}, {weights_dict['v_proj'].shape}"
            )

            q_proj_weight = weights_dict["q_proj"]
            k_proj_weight = weights_dict["k_proj"]
            v_proj_weight = weights_dict["v_proj"]

            print("Original QKV weights dimensions:")
            print("Q:", q_proj_weight.shape)
            print("K:", k_proj_weight.shape)
            print("V:", v_proj_weight.shape)

            # Reshape the weights to match the expected shape
            q_proj_weight_reshaped = q_proj_weight.reshape(-1, hidden_size)
            k_proj_weight_reshaped = k_proj_weight.reshape(-1, hidden_size // n_head)
            v_proj_weight_reshaped = v_proj_weight.reshape(-1, hidden_size // n_head)
            # q_proj_weight_reshaped = q_proj_weight.reshape(k_proj_weight_reshaped.shape[0], -1)

            print("Reshaped QKV weights dimensions:")
            print("Q:", q_proj_weight_reshaped.shape)
            print("K:", k_proj_weight_reshaped.shape)
            print("V:", v_proj_weight_reshaped.shape)

            combined_qkv = np.concatenate(
                [
                    q_proj_weight_reshaped,
                    k_proj_weight_reshaped,
                    v_proj_weight_reshaped,
                ],
                axis=1,
            )
            qkv_weight_name = qkv_name + ".weight"
            param_shape = model.state_dict()[qkv_weight_name].shape
            print(
                f"param shape expected to be {param_shape}, qkv weights combined with weights size {combined_qkv.shape}"
            )

            model.state_dict()[qkv_weight_name].copy_(torch.from_numpy(combined_qkv))
            print(f"Assigned combined QKV weights to {qkv_weight_name}.")
