import numpy as np
import os, torch, argparse, sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "peft"))
from alignment.align_test_utils import *
from transformers import AutoConfig
from tqdm import tqdm

class AlignmentTest:
    def __init__(self, hf_config, tp_degree=1):
        raise NotImplementedError()
    def check_weights_alignment(self):
        raise NotImplementedError()
    def check_fwd_pass(self):
        raise NotImplementedError()
    def check_bwd_pass(self):
        raise NotImplementedError()
    def check_step(self, step_idx, learning_rate=0.001):
        raise NotImplementedError()

class LllamaAlignmentTest(AlignmentTest):
    def __init__(self, hf_config, tp_degree=1):
        self.hf_config = hf_config
        self.num_layers = self.hf_config.num_hidden_layers
        self.hidden_size = self.hf_config.hidden_size
        self.intermediate_size = self.hf_config.intermediate_size
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = self.hf_config.num_key_value_heads
        self.projsize = self.hidden_size // self.num_attention_heads
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
            return f_version
        
        def get_hf_tensor(hf_tensor_name, tensor_comparison_idx):
            hf_tensor_filename = f"{hf_tensor_name}.{tensor_comparison_idx.hf_tensor_type}_{tensor_comparison_idx.hf_tensor_idx}"
            hf_tensor_path = os.path.join(hf_fwd_folder, hf_tensor_filename)

            if not os.path.isfile(hf_tensor_path):
                raise FileNotFoundError(f"File '{hf_tensor_path}' not found")
            print("loading hf tensor: ", hf_tensor_filename)
            hf_tensor = torch.load(hf_tensor_path, map_location='cpu')
            if hf_tensor_name == "embed_tokens":
                self.num_tokens = hf_tensor.shape[1]
            return hf_tensor
        
        def get_ff_tensor(ff_tensor_name, tensor_comparison_idx, hf_shape, tp_type=TPType.REPLICATE):
            ff_tensor_suffix = f".{tensor_comparison_idx.ff_tensor_type}" if len(tensor_comparison_idx.ff_tensor_type) > 0 else ""
            ff_tensor_idx_suffix = f"_{tensor_comparison_idx.ff_tensor_idx}" if tensor_comparison_idx.ff_tensor_idx is not None else ""
            ff_tensor_filename = f"{ff_tensor_name}{ff_tensor_suffix}{ff_tensor_idx_suffix}"
            ff_tensor_path = os.path.join(ff_fwd_folder, ff_tensor_filename)
            if not os.path.isfile(ff_tensor_path):
                raise FileNotFoundError(f"File '{ff_tensor_path}' not found")

            print("loading ff tensor: ", ff_tensor_filename)
            ff_shape = list(hf_shape)[::-1]
            if tp_type == TPType.PARTITION:
                ff_shape[0] //= self.tp_degree
            
            if "layers.0.embed_tokens.input_0" in ff_tensor_path:
                # get number of tokens
                ff_tensor = np.loadtxt(ff_tensor_path, delimiter=',')
                self.ff_batch_size = ff_tensor.shape[0]

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

        def compare(hf_tensor, ff_tensor, label="", additional_ff_tensor=None, tolerance=1e-2):
            ff_tensor = ff_tensor.to(hf_tensor.dtype)
            hf_tensor = hf_tensor.T
            if additional_ff_tensor is not None:
                additional_ff_tensor = additional_ff_tensor.to(hf_tensor.dtype)
                ff_tensor = ff_tensor - additional_ff_tensor
            try:
                # torch.testing.assert_close(hf_tensor, ff_tensor, rtol=1.3e-6, atol=tolerance)
                if not np.allclose(hf_tensor.detach().numpy(), ff_tensor.detach().numpy(), atol=tolerance):
                    mismatches = np.where(~np.isclose(hf_tensor.detach().numpy(), ff_tensor.detach().numpy(), atol=tolerance))[0]
                    print(f"Pct mismatch {label}: {100.0*(np.prod(mismatches.shape) / ff_tensor.numel()):.3f}%")
                    assert(np.prod(mismatches.shape) <= .05 * ff_tensor.numel())
            except Exception as e:
                print(f"Error in comparison {label}:\n{e}\n")
                print("HF tensor:")
                print(hf_tensor.squeeze())
                print(hf_tensor.shape)
                print("FF tensor:")
                print(ff_tensor.squeeze())
                print(ff_tensor.shape)
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
            compare(hf_tensor, ff_tensor, label=f"Input layernorm {i} input")
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Input layernorm {i} output")

            # Attention
            hf_tensor_name = f"layers.{i}.self_attn.o_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            # the raw attention result, w/o o_proj. This is the output of senf_attn of FF and the input of o_proj in HF
            output_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            # ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            # TP for self-attn partitions the attention heads across TP workers
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            print("comparing attention tensor: ", hf_tensor_name, " and ", ff_tensor_name)
            compare(hf_tensor, ff_tensor, label=f"Attention {i} output")
            
            # Post-attention layernorm
            hf_tensor_name = f"layers.{i}.post_attention_layernorm"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Post-attention layernorm {i} output")

            # W1 (gate_proj)
            hf_tensor_name = f"layers.{i}.mlp.gate_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W1 {i} output")

            # W3 (up_proj)
            hf_tensor_name = f"layers.{i}.mlp.up_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W3 {i} output")

            # W2 (down_proj)
            hf_tensor_name = f"layers.{i}.mlp.down_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_down_proj_out = get_hf_tensor(hf_tensor_name, output_comparison)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W2 {i} input")

            hf_down_proj_in = hf_tensor.clone()
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_down_proj_out = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
        
        # Norm
        hf_tensor_name = "norm"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Norm output")

        # LM head
        hf_tensor_name = "lm_head"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
        compare(hf_tensor, ff_tensor, label="LM head input")
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
        compare(hf_tensor, ff_tensor, label="LM head output")

class OPTAlignmentTest(AlignmentTest):
    def __init__(self, hf_config, tp_degree=1):
        self.hf_config = hf_config
        self.num_layers = self.hf_config.num_hidden_layers
        self.hidden_size = self.hf_config.hidden_size
        self.intermediate_size = self.hf_config.ffn_dim
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.projsize = self.hidden_size // self.num_attention_heads
        self.tp_degree = tp_degree

        self.num_tokens = None
        self.ff_batch_size = None
    
    def check_weights_alignment(self):
        def convert_hf_filename_to_ff(hf_filename):
            if hf_filename == "lm_head.weight":
                f_version = f"layers.{self.num_layers-1}.lm_head.weight_0"
            elif hf_filename == "final_layer_norm.weight":
                f_version = f"layers.{self.num_layers-1}.final_layer_norm.weight_0"
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
            if "lm_head" in ff_weight_name or "final_layer_norm" in ff_weight_name:
                return 1
            if "fc1" in ff_weight_name:
                return 1
            elif "fc2" in ff_weight_name:
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
            if hf_filename == "embed_tokens" or hf_filename == "embed_positions":
                f_version = f"layers.0.{hf_filename}"
            elif hf_filename == "lm_head" or hf_filename == "final_layer_norm":
                f_version = f"layers.{self.num_layers-1}.{hf_filename}"
            else:
                assert hf_filename.startswith("layers.")
                layernum = hf_filename.split("layers.")[1].split(".")[0]
                f_version = f"layers.{layernum}."
                f_version += hf_filename.replace(".base_layer", "").replace(".default", "")
                # right now, attention in flexflow is done with a single operator, so there is a single output file without the projection suffix
                f_version = f_version.replace(".q_proj", "").replace(".k_proj", "").replace(".v_proj", "").replace(".o_proj", "")
            return f_version
        
        def get_hf_tensor(hf_tensor_name, tensor_comparison_idx):
            hf_tensor_filename = f"{hf_tensor_name}.{tensor_comparison_idx.hf_tensor_type}_{tensor_comparison_idx.hf_tensor_idx}"
            hf_tensor_path = os.path.join(hf_fwd_folder, hf_tensor_filename)

            if not os.path.isfile(hf_tensor_path):
                raise FileNotFoundError(f"File '{hf_tensor_path}' not found")
            print("loading hf tensor: ", hf_tensor_filename)
            hf_tensor = torch.load(hf_tensor_path, map_location='cpu')
            if hf_tensor_name == "embed_tokens":
                self.num_tokens = hf_tensor.shape[1]
            return hf_tensor
        
        def get_ff_tensor(ff_tensor_name, tensor_comparison_idx, hf_shape, tp_type=TPType.REPLICATE):
            ff_tensor_suffix = f".{tensor_comparison_idx.ff_tensor_type}" if len(tensor_comparison_idx.ff_tensor_type) > 0 else ""
            ff_tensor_idx_suffix = f"_{tensor_comparison_idx.ff_tensor_idx}" if tensor_comparison_idx.ff_tensor_idx is not None else ""
            ff_tensor_filename = f"{ff_tensor_name}{ff_tensor_suffix}{ff_tensor_idx_suffix}"
            ff_tensor_path = os.path.join(ff_fwd_folder, ff_tensor_filename)
            if not os.path.isfile(ff_tensor_path):
                raise FileNotFoundError(f"File '{ff_tensor_path}' not found")

            print("loading ff tensor: ", ff_tensor_filename)
            ff_shape = list(hf_shape)[::-1]
            if tp_type == TPType.PARTITION:
                ff_shape[0] //= self.tp_degree
            
            if "layers.0.embed_tokens.input_0" in ff_tensor_path:
                # get number of tokens
                ff_tensor = np.loadtxt(ff_tensor_path, delimiter=',')
                self.ff_batch_size = ff_tensor.shape[0]

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

        def compare(hf_tensor, ff_tensor, label="", additional_ff_tensor=None, tolerance=1e-2):
            ff_tensor = ff_tensor.to(hf_tensor.dtype)
            hf_tensor = hf_tensor.T
            if additional_ff_tensor is not None:
                additional_ff_tensor = additional_ff_tensor.to(hf_tensor.dtype)
                ff_tensor = ff_tensor - additional_ff_tensor
            try:
                # torch.testing.assert_close(hf_tensor, ff_tensor, rtol=1.3e-6, atol=tolerance)
                if not np.allclose(hf_tensor.detach().numpy(), ff_tensor.detach().numpy(), atol=tolerance):
                    mismatches = np.where(~np.isclose(hf_tensor.detach().numpy(), ff_tensor.detach().numpy(), atol=tolerance))[0]
                    print(f"Pct mismatch {label}: {100.0*(np.prod(mismatches.shape) / ff_tensor.numel()):.3f}%")
                    assert(np.prod(mismatches.shape) <= .05 * ff_tensor.numel())
            except Exception as e:
                print(f"Error in comparison {label}:\n{e}\n")
                print("HF tensor:")
                print(hf_tensor.squeeze())
                print(hf_tensor.shape)
                print("FF tensor:")
                print(ff_tensor.squeeze())
                print(ff_tensor.shape)
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

        # Positional embedding layer
        hf_tensor_name = "embed_positions"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Position Embedding output")
        
        # Transformers blocks
        for i in range(self.num_layers):
            # Input laye norm
            hf_tensor_name = f"layers.{i}.self_attn_layer_norm"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Self attention layernorm {i} input")
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Self attention layernorm {i} output")

            # Attention
            hf_tensor_name = f"layers.{i}.self_attn.out_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name.replace(".out_proj", ".o_proj"))
            # the raw attention result, w/o o_proj. This is the output of senf_attn of FF and the input of o_proj in HF
            output_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            # ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            # TP for self-attn partitions the attention heads across TP workers
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            print("comparing attention tensor: ", hf_tensor_name, " and ", ff_tensor_name)
            compare(hf_tensor, ff_tensor, label=f"Attention {i} output")
            
            # Post-attention layernorm
            hf_tensor_name = f"layers.{i}.add_bias_residual_layer_norm"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
            compare(hf_tensor, ff_tensor, label=f"Add bias residual layernorm {i} output")

            # W1 (gate_proj)
            hf_tensor_name = f"layers.{i}.fc1"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"FC1 {i} output")

            # W2 (down_proj)
            hf_tensor_name = f"layers.{i}.fc2"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_down_proj_out = get_hf_tensor(hf_tensor_name, output_comparison)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"FC2 {i} input")

            hf_down_proj_in = hf_tensor.clone()
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_down_proj_out = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
        
        # Norm
        hf_tensor_name = "final_layer_norm"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=1)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Final layer norm output")

        # LM head
        hf_tensor_name = "lm_head"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
        compare(hf_tensor, ff_tensor, label="LM head input")
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
        compare(hf_tensor, ff_tensor, label="LM head output")

parser = argparse.ArgumentParser(description='Argument Parser Example') 
# Adding arguments
parser.add_argument('-m', '--model-name', type=str, default="goliaro/llama-160m-lora", help='Name of the model')
parser.add_argument('-n', '--num-steps', type=int, default=1, help='Number of decoding steps')
parser.add_argument('-tp', '--tensor-parallelism-degree', type=int, default=1, help='The tensor parallelism degree used when running FlexFlow')

# Parse the arguments from command line
args = parser.parse_args()

if __name__ == "__main__":
    hf_config = AutoConfig.from_pretrained(args.model_name)
    alignment_class = None
    if hf_config.architectures[0] == "LlamaForCausalLM":
        alignment_class = LllamaAlignmentTest(hf_config, tp_degree=args.tensor_parallelism_degree)
    elif hf_config.architectures[0] == "OPTForCausalLM":
        alignment_class = OPTAlignmentTest(hf_config, tp_degree=args.tensor_parallelism_degree)
    
    # alignment_class.check_weights_alignment()
    for i in range(args.num_steps):
        alignment_class.check_fwd_pass(i)
