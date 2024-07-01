import numpy as np
import os, torch, argparse
from alignment.align_test_utils import *
from transformers import AutoConfig
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass


def get_model_config(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model config for '{model_name}': {e}")
        config = None
    return config

class AlignmentTest:
    def __init__(self, model_name, tp_degree=1):
        raise NotImplementedError()
    def check_weights_alignment(self):
        raise NotImplementedError()
    def check_fwd_pass(self):
        raise NotImplementedError()
    def check_bwd_pass(self):
        raise NotImplementedError()
    def check_step(self, step_idx):
        raise NotImplementedError()

class LllamaAlignmentTest(AlignmentTest):
    def __init__(self, model_name, tp_degree=1):
        self.model_name = model_name
        self.hf_config = get_model_config(model_name)
        self.num_layers = self.hf_config.num_hidden_layers
        self.hidden_size = self.hf_config.hidden_size
        self.intermediate_size = self.hf_config.intermediate_size
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.tp_degree = tp_degree

        self.num_tokens = None
        self.ff_batch_size = None
    

    def check_weights_alignment(self):
        def convert_hf_filename_to_ff(hf_filename):
            if hf_filename == "lm_head.weight":
                f_version = f"layers.{self.num_layers-1}.lm_head.weight_0"
            elif hf_filename == "norm.weight":
                f_version = f"layers.{self.num_layers-1}.norm.weight_0"
            else:
                f_version = ""
                if hf_filename.startswith("layers."):
                    layernum = hf_filename.split("layers.")[1].split(".")[0]
                    f_version += f"layers.{layernum}."
                f_version += hf_filename.replace(".base_layer", "").replace(".default", "")
                # compute weight index, then rename lora if needed if needed
                weight_index="0"
                if "lora_A" in f_version:
                    weight_index="A"
                elif "lora_B" in f_version:
                    weight_index="B"
                f_version = f_version.replace("lora_A", "lora").replace("lora_B", "lora")
                if f_version.endswith(".weight"):
                    if weight_index == "0":
                        f_version += f"_{weight_index}"
                    else:
                        f_version += f"_{weight_index}.original"
                elif f_version.endswith(".gradient"):
                    prefix = f_version.split(".gradient")[0]
                    f_version = prefix + f".weight_{weight_index}.gradient"
            return f_version
        def get_tp_partition_dim(ff_weight_name) -> int:
            # MLP layers split the intermediate size dimension
            # gate_proj, up_proj: [hidden_size, intermediate_size]
            # down_proj: [intermediate_size, hidden_size]
            if self.tp_degree == 1:
                return -1
            if "lora.weight_B" in ff_weight_name:
                return -1
            if "lm_head" in ff_weight_name or "norm" in ff_weight_name:
                return 1
            if "gate_proj" in ff_weight_name or "up_proj" in ff_weight_name:
                return 1
            elif "down_proj" in ff_weight_name:
                return 0
            else:
                return -1
        print("-- Weights alignment --")
        hf_weights_folder = os.path.join(hf_path, "weights", "step_0")
        ff_weights_folder = os.path.join(ff_path, "weights", "step_0", "shard_0")
        files_list = os.listdir(hf_weights_folder)
        for hf_weight_name in tqdm(sorted(files_list)):
            if hf_weight_name.endswith(".weight"):
                ff_weight_name = convert_hf_filename_to_ff(hf_weight_name)
                # print(hf_weight_name, ff_weight_name)
                hf_w_path = os.path.join(hf_weights_folder, hf_weight_name)
                ff_w_path = os.path.join(ff_weights_folder, ff_weight_name)
                if not os.path.isfile(hf_w_path):
                    print(f"File '{hf_w_path}' not found")
                if not os.path.isfile(ff_w_path):
                    print(f"File '{ff_w_path}' not found")
                assert(os.path.isfile(hf_w_path))
                assert(os.path.isfile(ff_w_path))

                # 1. get shape of hf weight
                hf_weight = torch.load(hf_w_path, map_location='cpu')
                hf_weigth_shape = hf_weight.shape
                ff_partition_dim = get_tp_partition_dim(ff_weight_name)
                ff_weigth_shape = list(hf_weigth_shape)[::-1]
                if ff_partition_dim >= 0:
                    ff_weigth_shape[ff_partition_dim] //= self.tp_degree
                
                # 2. handle flexflow shards in case of tensor parallelism
                ff_weights = [load_ff_tensor(ff_w_path.replace("shard_0", f"shard_{tp_idx}"), ff_weigth_shape) for tp_idx in range(self.tp_degree)]
                if self.tp_degree > 1:
                    if ff_partition_dim >= 0:
                        ff_weight = np.concatenate(ff_weights, axis=ff_partition_dim)
                    else:
                        assert(are_np_arrays_identical(ff_weights))
                        ff_weight = ff_weights[0]
                else:
                    ff_weight = ff_weights[0]
                ff_weight = torch.from_numpy(ff_weight).to(hf_weight.dtype)
                
                # check equivalence
                try:
                    torch.testing.assert_close(ff_weight, hf_weight.T)
                except Exception as e:
                    print(f"Error comparing {ff_w_path} weight to {hf_w_path}:\n{e}\n")
                    raise e
    
    def check_fwd_pass(self, step_idx=0):
        hf_fwd_folder = os.path.join(hf_path, "fwd", f"step_{step_idx}")
        ff_fwd_folder = os.path.join(ff_path, "fwd", f"step_{step_idx}", "shard_0")
        
        def convert_hf_filename_to_ff(hf_filename):
            if hf_filename == "embed_tokens":
                f_version = f"layers.0.embed_tokens"
            elif hf_filename == "lm_head" or hf_filename == "norm":
                f_version = f"layers.{self.num_layers-1}.{hf_filename}"
            else:
                assert hf_filename.startswith("layers.")
                layernum = hf_filename.split("layers.")[1].split(".")[0]
                f_version = f"layers.{layernum}."
                f_version += hf_filename.replace(".base_layer", "").replace(".default", "")
                # right now, attention in flexflow is done with a single operator, so there is a single output file without the projection suffix
                f_version = f_version.replace(".q_proj", "").replace(".k_proj", "").replace(".v_proj", "").replace(".o_proj", "")
                # lora in HuggingFace is split into A and B operators, in FF we use a single operator.
                f_version = f_version.replace("lora_A", "lora").replace("lora_B", "lora")
            return f_version
        
        class TPType(Enum):
            REPLICATE = 0
            PARTITION = 1
            TO_REDUCE = 2
        
        def replace_value(lst, old_value, new_value):
            occurrences = lst.count(old_value)
            if occurrences == 0:
                raise ValueError(f"Value {old_value} not found in the list.")
            elif occurrences > 1:
                raise ValueError(f"Multiple instances of {old_value} found in the list.")
            else:
                index = lst.index(old_value)
                lst[index] = new_value
                return lst
        
        def truncate_dimension(tensor, old_dim, new_dim):
            # Check if old_dim appears exactly once in the tensor's shape
            shape = tensor.shape
            dim_occurrences = shape.count(old_dim)
            
            if dim_occurrences == 0:
                raise ValueError(f"Dimension {old_dim} not found in the tensor shape.")
            elif dim_occurrences > 1:
                raise ValueError(f"Multiple instances of dimension {old_dim} found in the tensor shape.")
            
            # Check if new_dim is less than or equal to old_dim
            if new_dim > old_dim:
                raise ValueError(f"New dimension ({new_dim}) must be less than or equal to old dimension ({old_dim}).")
            
            # Find the index of the dimension to truncate
            dim_index = shape.index(old_dim)
            
            # Create a slice object for truncation
            slices = [slice(None)] * len(shape)
            slices[dim_index] = slice(0, new_dim)
            
            # Truncate the tensor
            truncated_tensor = tensor[tuple(slices)]
            
            return truncated_tensor
        
        @dataclass
        class TensorComparisonIdxs:
            hf_tensor_type: str
            ff_tensor_type: str
            hf_tensor_idx: int
            ff_tensor_idx: int
        
        def get_hf_tensor(hf_tensor_name, tensor_comparison_idx):
            hf_tensor_filename = f"{hf_tensor_name}.{tensor_comparison_idx.hf_tensor_type}_{tensor_comparison_idx.hf_tensor_idx}"
            hf_tensor_path = os.path.join(hf_fwd_folder, hf_tensor_filename)
            print(hf_tensor_path)
            if not os.path.isfile(hf_tensor_path):
                raise FileNotFoundError(f"File '{hf_tensor_path}' not found")
            hf_tensor = torch.load(hf_tensor_path, map_location='cpu')
            if hf_tensor_name == "embed_tokens":
                self.num_tokens = hf_tensor.shape[1]
            return hf_tensor
        
        def get_ff_tensor(ff_tensor_name, tensor_comparison_idx, hf_shape, tp_type=TPType.REPLICATE):
            ff_tensor_filename = f"{ff_tensor_name}.{tensor_comparison_idx.ff_tensor_type}_{tensor_comparison_idx.ff_tensor_idx}"
            ff_tensor_path = os.path.join(ff_fwd_folder, ff_tensor_filename)
            if not os.path.isfile(ff_tensor_path):
                raise FileNotFoundError(f"File '{ff_tensor_path}' not found")

            ff_shape = list(hf_shape)[::-1]
            if tp_type == TPType.PARTITION:
                ff_shape[0] //= self.tp_degree
            
            if "layers.0.embed_tokens.input_0" in ff_tensor_path:
                # get number of tokens
                ff_tensor = np.loadtxt(ff_tensor_path, delimiter=',')
                self.ff_batch_size = ff_tensor.shape[0]
            print(ff_tensor_path)
            ff_shape = replace_value(ff_shape, self.num_tokens, self.ff_batch_size)
            ff_tensors = [load_ff_tensor(ff_tensor_path.replace("shard_0", f"shard_{tp_idx}"), ff_shape) for tp_idx in range(self.tp_degree)]
            if self.tp_degree > 1:
                # if replicate, check that they are identical
                if tp_type == TPType.REPLICATE:
                    assert(are_np_arrays_identical(ff_tensors))
                    ff_tensor = ff_tensors[0]
                # if partition, concatenate along the partition dimension
                elif tp_type == TPType.PARTITION:
                    ff_tensor = np.concatenate(ff_tensors, axis=0)
                # if to_reduce, sum along the partition dimension
                elif tp_type == TPType.TO_REDUCE:
                    ff_tensor = np.sum(ff_tensors, axis=0)
            else:
                ff_tensor = ff_tensors[0]
            ff_tensor = torch.from_numpy(ff_tensor)
            ff_tensor = truncate_dimension(ff_tensor, self.ff_batch_size, self.num_tokens)
            return ff_tensor

        def compare(hf_tensor, ff_tensor, label="", additional_ff_tensor=None, tolerance=1e-5):
            ff_tensor = ff_tensor.to(hf_tensor.dtype)
            hf_tensor = hf_tensor.T
            if additional_ff_tensor is not None:
                additional_ff_tensor = additional_ff_tensor.to(hf_tensor.dtype)
                ff_tensor = ff_tensor - additional_ff_tensor
            try:
                torch.testing.assert_close(hf_tensor, ff_tensor, rtol=1.3e-6, atol=tolerance)
            except Exception as e:
                print(f"Error in comparison {label}:\n{e}\n")
                print("HF tensor:")
                print(hf_tensor.squeeze())
                print("FF tensor:")
                print(ff_tensor.squeeze())
                raise e

        print(f"-- FWD pass {step_idx}--")
        
        # Embedding layer
        hf_tensor_name = "embed_tokens"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Embedding input")
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Embedding output")
        
        # Transformers blocks
        for i in range(self.num_layers):
            # Input laye norm
            hf_tensor_name = f"layers.{i}.input_layernorm"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            if i == 0:
                input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
                output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            else:
                input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
                output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Input layernorm {i} input", tolerance=1e-4)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Input layernorm {i} output", tolerance=1e-4)

            # Attention
            hf_tensor_name = f"layers.{i}.self_attn.o_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            compare(hf_tensor, ff_tensor, label=f"Attention {i} output", tolerance=1e-4)
            
            # Post-attention layernorm
            hf_tensor_name = f"layers.{i}.post_attention_layernorm"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Post-attention layernorm {i} output", tolerance=1e-4)

            # W1 (gate_proj)
            hf_tensor_name = f"layers.{i}.mlp.gate_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W1 {i} output", tolerance=1e-4)

            # W3 (up_proj)
            hf_tensor_name = f"layers.{i}.mlp.up_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W3 {i} output", tolerance=1e-4)

            # W2 (down_proj)
            hf_tensor_name = f"layers.{i}.mlp.down_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W2 {i} input", tolerance=1e-4)

            hf_down_proj_in = hf_tensor.clone()
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_down_proj_out = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)

            # LoRA_A
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_A.default"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"LoRA_A {i} input", tolerance=1e-4)
            torch.testing.assert_close(hf_down_proj_in, hf_tensor, rtol=1.3e-6, atol=1e-5)

            # LoRA intermediate (HF only)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_lora_A_out = get_hf_tensor(hf_tensor_name, output_comparison)
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_B.default"
            hf_lora_B_in = get_hf_tensor(hf_tensor_name, input_comparison)
            torch.testing.assert_close(hf_lora_A_out, hf_lora_B_in, rtol=1.3e-6, atol=1e-5)

            # LoRA_B
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_B.default"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            compare(hf_tensor, ff_tensor, additional_ff_tensor=ff_down_proj_out, label=f"LoRA_B {i} output", tolerance=1e-4)
        
        # Norm
        hf_tensor_name = "norm"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Norm output", tolerance=1e-4)

        # LM head
        hf_tensor_name = "lm_head"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
        compare(hf_tensor, ff_tensor, label="LM head output", tolerance=1e-4)

        
    def check_bwd_pass(self):
        raise NotImplementedError()
    def check_step(self, step_idx):
        raise NotImplementedError()

def check_weights_alignment(num_layers=12):
    print("-- Weights alignment --")
    files_list = os.listdir(hf_path)

    for f in sorted(files_list):
        if f.endswith(".weight"):
            if "self_attn" in f:
                continue
            f_version = convert_hf_filename_to_ff_filename(f, num_layers=num_layers)
            # print(f, f_version)
            hf_w_path = os.path.join(hf_path, f)
            ff_w_path = os.path.join(ff_path, f_version)
            if not os.path.isfile(hf_w_path):
                print(f"File '{hf_w_path}' not found")
            if not os.path.isfile(ff_w_path):
                print(f"File '{ff_w_path}' not found")
            assert(os.path.isfile(hf_w_path))
            assert(os.path.isfile(ff_w_path))
            # print("\t", os.path.isfile(hf_w_path), os.path.isfile(ff_w_path))
            # print("\t", ff_w_path)

            # check equivalence
            try:
                compare_tensors(hf_w_path, ff_w_path, tolerance=1e-5)
            except Exception as e:
                print(f"Error comparing {ff_w_path} weight to {hf_w_path}:\n{e}\n")
                raise e

def check_llama_fwd_pass(hf_config, tot_num_layers = 12, step_idx=0):
    print(f"-- FWD pass {step_idx}--")
    
    # Transfomer head
    hf_embed_input= f"{hf_path}/fwd_step_{step_idx}_embed_tokens.input_0"
    ff_embed_input = f"{ff_path}/fwd_step_{step_idx}_layers_0_embed_tokens_shard_0_input_0"
    compare_tensors(hf_embed_input, ff_embed_input)
    hf_embed_output = f"{hf_path}/fwd_step_{step_idx}_embed_tokens.output_0"
    ff_embed_output = f"{ff_path}/fwd_step_{step_idx}_layers_0_embed_tokens_shard_0_output_0"
    compare_tensors(hf_embed_output, ff_embed_output)

    # Transformers blocks
    for i in range(tot_num_layers):
        hf_input_ln_in = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.input_layernorm.input_0"
        ff_input_ln_in = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_input_0"
        if i > 0:
            ff_input_ln_in = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_output_0"
        compare_tensors(hf_input_ln_in, ff_input_ln_in, tolerance=1e-5)
        hf_input_ln_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.input_layernorm.output_0"
        ff_input_ln_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_output_0"
        if i > 0:
            ff_input_ln_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_output_1"
        compare_tensors(hf_input_ln_out, ff_input_ln_out, tolerance=1e-5)
        hf_attn_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.self_attn.o_proj.output_0"
        ff_attn_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_output_0"
        compare_tensors(hf_attn_out, ff_attn_out, tolerance=1e-5)
        hf_ffn_norm_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.post_attention_layernorm.output_0"
        ff_ffn_norm_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_output_1"
        compare_tensors(hf_ffn_norm_out, ff_ffn_norm_out, tolerance=1e-5)
        # w1
        hf_gate_proj_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.gate_proj.output_0"
        ff_gate_proj_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_output_0"
        compare_tensors(hf_gate_proj_out, ff_gate_proj_out, tolerance=1e-5)
        # w3
        hf_up_proj_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.up_proj.output_0" 
        ff_up_proj_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.up_proj_shard_0_output_0"
        compare_tensors(hf_up_proj_out, ff_up_proj_out, tolerance=1e-5)
        # w2
        hf_down_proj_in = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.down_proj.input_0"
        hf_down_proj_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.down_proj.output_0"
        ff_down_proj_in = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj_shard_0_input_0"
        ff_down_proj_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj_shard_0_output_0"
        compare_tensors(hf_down_proj_in, ff_down_proj_in)
        # compare_tensors(hf_down_proj_out, ff_down_proj_out)
        # LORA input
        hf_lora_A_in = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_A.default.input_0"
        ff_lora_A_in = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_input_0"
        compare_hf_tensors(hf_down_proj_in, hf_lora_A_in)
        compare_tensors(hf_lora_A_in, ff_lora_A_in)
        # LORA weights
        hf_lora_A_weight_fp = f"{hf_path}/layers.{i}.mlp.down_proj.lora_A.default.weight"
        ff_lora_A_weight_fp = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_A"
        compare_tensors(hf_lora_A_weight_fp, ff_lora_A_weight_fp)
        hf_lora_B_weight_fp = f"{hf_path}/layers.{i}.mlp.down_proj.lora_B.default.weight"
        ff_lora_B_weight_fp = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_B"
        compare_tensors(hf_lora_B_weight_fp, ff_lora_B_weight_fp)
        # LORA intermediate hf
        hf_lora_A_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_A.default.output_0"
        hf_lora_B_in = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_B.default.input_0"
        compare_hf_tensors(hf_lora_A_out, hf_lora_B_in)
        # LORA output
        hf_lora_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_B.default.output_0"
        ff_lora_out = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_output_0"
        # compare_tensors(hf_lora_out, ff_lora_out)
        # compare_flexflow_tensors(ff_down_proj_out, ff_lora_out)
        # compare_tensors(hf_down_proj_out, ff_lora_out)
        compare_tensors_difference(hf_lora_out, ff_lora_out, ff_down_proj_out)
        

    # After last layer only
    hf_norm_out = f"{hf_path}/fwd_step_{step_idx}_norm.output_0"
    ff_norm_out = f"{ff_path}/fwd_step_{step_idx}_layers_{tot_num_layers-1}_norm_shard_0_output_1"
    compare_tensors(hf_norm_out, ff_norm_out, tolerance=1e-5)
    hf_lm_head_out = f"{hf_path}/fwd_step_{step_idx}_base_model.model.lm_head.output_0"
    ff_lm_head_out = f"{ff_path}/fwd_step_{step_idx}_layers_{tot_num_layers-1}_lm_head_shard_0_output_0"
    compare_tensors(hf_lm_head_out, ff_lm_head_out, tolerance=1e-5)

def check_llama_bwd_pass(hf_config, tot_num_layers = 12, step_idx=0):
    # ff_BWD_softmax_in = f"{ff_path}/model_0_bwd-step_{step_idx}_layer-num_100_layer-name_Softmax_shard-id_0_input_0"
    print("-- LM head --")
    hf_BWD_lm_head_out = f"{hf_path}/bwd_step_{step_idx}_base_model.model.lm_head.go_0"
    ff_BWD_lm_head_out = f"{ff_path}/bwd_step_{step_idx}_layers_{tot_num_layers-1}_lm_head_shard_0_output_0"
    compare_tensors(hf_BWD_lm_head_out, ff_BWD_lm_head_out, tolerance=1e-5)
    # compare weights
    # hf_lm_head_weight = f"{hf_path}/base_model.model.lm_head.weight"
    # ff_lm_head_weight = f"{ff_path}/fwd_step_{step_idx}_layers_{tot_num_layers-1}_output_shard_0_weight_0"
    # compare_tensors(hf_lm_head_weight, ff_lm_head_weight, tolerance=1e-5)
    hf_BWD_lm_head_in = f"{hf_path}/bwd_step_{step_idx}_base_model.model.lm_head.gi_0"
    ff_BWD_lm_head_in = f"{ff_path}/bwd_step_{step_idx}_layers_{tot_num_layers-1}_lm_head_shard_0_input_0"
    compare_tensors(hf_BWD_lm_head_in, ff_BWD_lm_head_in, tolerance=1e-5)
    # # Manually check the matmul
    # ff_tensor_out = np.loadtxt(ff_BWD_lm_head_out, delimiter=',')
    # ff_weight = np.loadtxt(ff_lm_head_weight, delimiter=',').reshape((4096,32000), order='F')
    # ff_tensor_out = ff_tensor_out[:32000*24].reshape((32000,24), order='F')
    # print(ff_tensor_out.shape)
    # print(ff_weight.shape)
    # print(np.matmul(ff_weight, ff_tensor_out))
    # compare_tensors(hf_BWD_lm_head_in, ff_BWD_lm_head_in)
    # ff_tensor = np.loadtxt(ff_tensor_filepath, delimiter=',')
    print("-- Final Norm --")
    hf_BWD_norm_out = f"{hf_path}/bwd_step_{step_idx}_norm.go_0"
    ff_BWD_norm_out = f"{ff_path}/bwd_step_{step_idx}_layers_{tot_num_layers-1}_norm_shard_0_output_0"
    compare_hf_tensors(hf_BWD_lm_head_in, hf_BWD_norm_out)
    compare_tensors(hf_BWD_norm_out, ff_BWD_norm_out)
    ff_BWD_norm_weight = f"{ff_path}/fwd_step_{step_idx}_layers_{tot_num_layers-1}_norm_shard_0_weight_0"
    hf_FWD_norm_weight = f"{hf_path}/norm.weight"
    compare_tensors(hf_FWD_norm_weight, ff_BWD_norm_weight, tolerance=1e-5)
    hf_BWD_norm_in = f"{hf_path}/bwd_step_{step_idx}_norm.gi_0"
    ff_BWD_norm_in = f"{ff_path}/bwd_step_{step_idx}_layers_{tot_num_layers-1}_norm_shard_0_input_1"
    compare_tensors(hf_BWD_norm_in, ff_BWD_norm_in, tolerance=1e-5)

    print("-- Transformers blocks --")
    for i in range(tot_num_layers-1, -1, -1):
        # HuggingFace filepaths
        hf_BWD_norm_in = f"{hf_path}/bwd_step_{step_idx}_norm.gi_0"
        hf_BWD_loraB_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_B.default.go_0"
        hf_BWD_loraB_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_B.default.gi_0"
        hf_BWD_loraA_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_A.default.go_0"
        hf_BWD_loraA_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.down_proj.lora_A.default.gi_0"
        hf_loraA_weight = f"{hf_path}/layers.{i}.mlp.down_proj.lora_A.default.weight"
        hf_loraB_weight = f"{hf_path}/layers.{i}.mlp.down_proj.lora_B.default.weight"
        hf_BWD_w2_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.down_proj.go_0"
        hf_BWD_w2_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.down_proj.gi_0"
        hf_w2_weight = f"{hf_path}/layers.{i}.mlp.down_proj.base_layer.weight"
        hf_BWD_w3_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.up_proj.go_0"
        hf_BWD_w3_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.up_proj.gi_0"
        hf_BWD_w1_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.gate_proj.go_0"
        hf_BWD_w1_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.gate_proj.gi_0"
        hf_BWD_act_fn_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.mlp.act_fn.gi_0"
        hf_BWD_ffn_norm_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.post_attention_layernorm.go_0"
        hf_BWD_ffn_norm_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.post_attention_layernorm.gi_0"
        hf_BWD_attn_out_out = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.self_attn.o_proj.go_0"
        hf_BWD_attn_q_in = f"{hf_path}/bwd_step_{step_idx}_layers.11.self_attn.q_proj.gi_0"
        hf_FWD_w1_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.gate_proj.output_0"
        hf_FWD_w3_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.up_proj.output_0"
        hf_FWD_act_fn_out = f"{hf_path}/fwd_step_{step_idx}_layers.{i}.mlp.act_fn.output_0"
        hf_BWD_attn_oproj_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.self_attn.o_proj.gi_0"
        hf_attn_qproj_weight = f"{hf_path}/layers.{i}.self_attn.q_proj.weight"
        hf_attn_kproj_weight = f"{hf_path}/layers.{i}.self_attn.k_proj.weight"
        hf_attn_vproj_weight = f"{hf_path}/layers.{i}.self_attn.v_proj.weight"
        hf_attn_oproj_weight = f"{hf_path}/layers.{i}.self_attn.o_proj.weight"
        
        # FlexFlow filepaths
        ff_BWD_w2_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj_shard_0_output_0"
        ff_BWD_w2_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj_shard_0_input_0"
        ff_BWD_w2_in_pre = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj_shard_0_pre_input_0"
        ff_w2_weight = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj_shard_0_weight_0"
        ff_BWD_ssm_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_SigmoidSiluMulti_shard_0_output_0"
        ff_BWD_ssm_in1 = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_SigmoidSiluMulti_shard_0_input_0"
        ff_BWD_ssm_in2 = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_SigmoidSiluMulti_shard_0_input_1"
        ff_BWD_w3_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.up_proj_shard_0_output_0"
        ff_BWD_w3_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.up_proj_shard_0_input_0"
        ff_BWD_lora_A_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_input_0"
        ff_BWD_lora_B_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_output_0"
        ff_lora_A_weight = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_A"
        ff_lora_B_weight = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_B"
        ff_BWD_w1_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_output_0"
        ff_BWD_w1_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_input_0"
        ff_BWD_w1_in_pre = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_pre_input_0"
        ff_BWD_ffn_norm_in1 = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_input_0"
        ff_BWD_ffn_norm_in2 = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_input_1"
        ff_BWD_ffn_norm_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_output_0"
        ff_BWD_attn_out = ff_path + f"/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_output_0"        
        ff_BWD_attn_o_proj_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_o_proj_in_grad"
        ff_attn_oproj_weight = f"{ff_path}/fwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_weight_0"

        # HuggingFace checks
        print("\nHuggingface checks:")
        if i == tot_num_layers-1:
            compare_hf_tensors(hf_BWD_norm_in, hf_BWD_loraB_out)
            compare_hf_tensors(hf_BWD_norm_in, hf_BWD_w2_out)
        compare_hf_tensors(hf_BWD_loraB_out, hf_BWD_w2_out)
        compare_hf_tensors(hf_BWD_loraB_in, hf_BWD_loraA_out)

        compare_hf_tensors(hf_BWD_act_fn_in, hf_BWD_w1_out)
        check_hf_sum_tensors(hf_BWD_ffn_norm_out, hf_BWD_w1_in, hf_BWD_w3_in)
        if i == tot_num_layers-1:
            check_hf_sum_tensors(hf_BWD_attn_out_out, hf_BWD_ffn_norm_in, hf_BWD_norm_in)

        # FlexFlow checks
        print("\nFlexFlow checks:")
        compare_flexflow_tensors(ff_BWD_w2_out, ff_BWD_lora_B_out)
        compare_flexflow_tensors(ff_BWD_w2_in_pre, ff_BWD_lora_A_in)
        compare_flexflow_tensors(ff_BWD_w2_in, ff_BWD_ssm_out)
        compare_flexflow_tensors(ff_BWD_ssm_in2, ff_BWD_w3_out)
        compare_flexflow_tensors(ff_BWD_ssm_in1, ff_BWD_w1_out)
        # compare_flexflow_tensors(ff_BWD_w1_in, ff_BWD_ffn_norm_out)
        compare_flexflow_tensors(ff_BWD_w1_in_pre, ff_BWD_w3_in)
        # compare_flexflow_tensors(ff_BWD_ffn_norm_in1, ff_BWD_ffn_norm_in2, max_len=24*768)
        
        # HF-FlexFlow checks
        print("\nHuggingface-FlexFlow checks:")
        print("-- W2 --")
        compare_tensors(hf_BWD_w2_out, ff_BWD_w2_out, tolerance=1e-5)
        compare_tensors(hf_w2_weight, ff_w2_weight, tolerance=1e-5)
        
        print("-- Lora --")
        compare_tensors(hf_loraA_weight, ff_lora_A_weight, tolerance=1e-5)
        compare_tensors(hf_loraB_weight, ff_lora_B_weight, tolerance=1e-5)

        compare_tensors(hf_BWD_loraB_out, ff_BWD_lora_B_out)
        compare_tensors(hf_BWD_loraA_in, ff_BWD_lora_A_in)
        
        print("-- W2/W1/W3 --")
        compare_tensors(hf_BWD_w2_in, ff_BWD_ssm_out)
        compare_tensors(hf_BWD_w2_in, ff_BWD_w2_in)
        compare_tensors(hf_BWD_w1_out, ff_BWD_w1_out)
        compare_tensors_difference(hf_BWD_w1_in, ff_BWD_w1_in, ff_BWD_w1_in_pre)
        compare_tensors(hf_BWD_w3_out, ff_BWD_w3_out)
        compare_tensors(hf_BWD_w3_in, ff_BWD_w3_in)
        compare_tensors(hf_BWD_w1_out, ff_BWD_w1_out)
        
        print("-- Attention --")
        num_tokens = 24
        hidden_size = 768
        qProjSize = 64
        num_heads = 12
        # Check output
        compare_tensors(hf_BWD_attn_out_out, ff_BWD_attn_out)
        
        # Check weights
        ff_attn_weight_tensor = np.loadtxt(ff_attn_oproj_weight, delimiter=',')
        ff_attn_qproj_weight_tensor = ff_attn_weight_tensor[:hidden_size*qProjSize*num_heads].reshape((hidden_size,qProjSize*num_heads), order = 'F')
        ff_attn_kproj_weight_tensor = ff_attn_weight_tensor[hidden_size*qProjSize*num_heads:2*hidden_size*qProjSize*num_heads].reshape((hidden_size,qProjSize*num_heads), order = 'F')
        ff_attn_vproj_weight_tensor = ff_attn_weight_tensor[2*hidden_size*qProjSize*num_heads:3*hidden_size*qProjSize*num_heads].reshape((hidden_size,qProjSize*num_heads), order = 'F')
        ff_attn_oproj_weight_tensor = ff_attn_weight_tensor[3*hidden_size*qProjSize*num_heads:].reshape((qProjSize*num_heads,hidden_size), order='F')
        
        hf_attn_qproj_weight_tensor = torch.load(hf_attn_qproj_weight).T.detach().cpu().numpy()
        hf_attn_kproj_weight_tensor = torch.load(hf_attn_kproj_weight).T.detach().cpu().numpy()
        hf_attn_vproj_weight_tensor = torch.load(hf_attn_vproj_weight).T.detach().cpu().numpy()
        hf_attn_oproj_weight_tensor = torch.load(hf_attn_oproj_weight).T.detach().cpu().numpy()
        
        assert(np.allclose(ff_attn_qproj_weight_tensor, hf_attn_qproj_weight_tensor, atol=1e-5))
        assert(np.allclose(ff_attn_kproj_weight_tensor, hf_attn_kproj_weight_tensor, atol=1e-5))
        assert(np.allclose(ff_attn_vproj_weight_tensor, hf_attn_vproj_weight_tensor, atol=1e-5))
        assert(np.allclose(ff_attn_oproj_weight_tensor, hf_attn_oproj_weight_tensor, atol=1e-5))

        # Compare attn outproj grad in tensors
        compare_tensors(hf_BWD_attn_oproj_in, ff_BWD_attn_o_proj_in)

        # Compare vproj grads
        hf_vproj_grads = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.self_attn.v_proj.go_0"
        ff_vproj_grads = ff_path + f"/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_v_proj_in_grad"
        hf_vproj_grads = torch.load(hf_vproj_grads).squeeze().detach().cpu().numpy()
        ff_vproj_grads = np.loadtxt(ff_vproj_grads, delimiter=',').reshape((num_tokens, qProjSize*num_heads), order='F')
        compare_loaded_tensors(hf_vproj_grads, ff_vproj_grads)

        # Compare kproj grads
        ff_kproj = ff_path + f"/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_devkproj"
        ff_kproj = np.loadtxt(ff_kproj, delimiter=',').reshape((num_tokens, qProjSize, num_heads), order = 'F')
        hf_kproj_grads = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.self_attn.k_proj.go_0"
        hf_kproj_grads = torch.load(hf_kproj_grads).squeeze()
        reshaped_tensor = hf_kproj_grads.view(24, 12, 64).transpose(1, 2).contiguous().detach().cpu().numpy()
        assert(np.allclose(ff_kproj, reshaped_tensor, atol=1e-2))
        print("Ok!")

        # Compare qproj grads
        hf_qproj_grads = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.self_attn.q_proj.go_0"
        hf_qproj_grads = torch.load(hf_qproj_grads).squeeze()
        reshaped_tensor = hf_qproj_grads.view(24, 12, 64).transpose(1, 2).contiguous().detach().cpu().numpy()
        ff_qproj = ff_path + f"/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_devQKVPRojArray"
        ff_qproj = np.loadtxt(ff_qproj, delimiter=',').reshape((num_tokens, qProjSize, num_heads, 3), order = 'F')[:,:,:,0]
        assert(np.allclose(ff_qproj, reshaped_tensor, atol=1e-2))
        print("Ok!")

        # Compare attn grad input 
        hf_attn_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.input_layernorm.go_0"
        ff_attn_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_attn_final_grad_in"
        compare_tensors(hf_attn_in, ff_attn_in)

        # compare input layernorm
        print("-- Input LayerNorm --")
        if i > 0:
            ff_input_ln_out = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_output_1"
            ff_attn_operator_in = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.self_attn_shard_0_input_0"
            compare_flexflow_tensors(ff_attn_operator_in, ff_input_ln_out)
            hf_input_ln_in = f"{hf_path}/bwd_step_{step_idx}_layers.{i}.input_layernorm.gi_0"
            ff_input_ln_in0 = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_input_0"
            ff_input_ln_in1 = f"{ff_path}/bwd_step_{step_idx}_layers_{i}_layers.{i}.input_layernorm_shard_0_input_1"
            compare_flexflow_tensors(ff_input_ln_in0, ff_input_ln_in1)
            if i > 1:
                compare_tensors(hf_input_ln_in, ff_input_ln_in0)
        

parser = argparse.ArgumentParser(description='Argument Parser Example') 
# Adding arguments
parser.add_argument('-m', '--model-name', type=str, default="JackFram/llama-160m", help='Name of the model')
parser.add_argument('-n', '--num-layers', type=int, default=12, help='Number of layers in the model')
parser.add_argument('-tp', '--tensor-parallelism-degree', type=int, default=1, help='The tensor parallelism degree used when running FlexFlow')

# Parse the arguments from command line
args = parser.parse_args()

if __name__ == "__main__":
    llama_alignment = LllamaAlignmentTest(args.model_name, tp_degree=args.tensor_parallelism_degree)
    # llama_alignment.check_weights_alignment()
    llama_alignment.check_fwd_pass()
    # hf_config = get_model_config(args.model_name)

    # check_weights_alignment(num_layers=args.num_layers)
    # n_steps=5
    # for i in range(1, n_steps):
    #     check_llama_fwd_pass(hf_config, tot_num_layers=args.num_layers, step_idx=i)
    #     check_llama_bwd_pass(hf_config, tot_num_layers=args.num_layers, step_idx=i)
