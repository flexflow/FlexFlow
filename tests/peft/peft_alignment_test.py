import numpy as np
import os, torch, argparse
from alignment.align_test_utils import *
from transformers import AutoConfig
from peft import PeftConfig
from tqdm import tqdm

class AlignmentTest:
    def __init__(self, model_name, tp_degree=1):
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
    def __init__(self, model_name, tp_degree=1):
        self.model_name = model_name
        self.peft_config = PeftConfig.from_pretrained(model_name)
        self.hf_config = AutoConfig.from_pretrained(self.peft_config.base_model_name_or_path)
        self.num_layers = self.hf_config.num_hidden_layers
        self.hidden_size = self.hf_config.hidden_size
        self.intermediate_size = self.hf_config.intermediate_size
        self.num_attention_heads = self.hf_config.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.projsize = self.hidden_size // self.num_attention_heads
        self.tp_degree = tp_degree
        self.lora_scaling_factor = self.peft_config.lora_alpha / self.peft_config.r

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

            # LoRA_A
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_A.default"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"LoRA_A {i} input")
            torch.testing.assert_close(hf_down_proj_in, hf_tensor, rtol=1.3e-6, atol=1e-5)

            # LoRA intermediate
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input", ff_tensor_type="input", hf_tensor_idx=0, ff_tensor_idx=0)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="low_rank_activation", hf_tensor_idx=0, ff_tensor_idx=None)
            hf_lora_A_out = get_hf_tensor(hf_tensor_name, output_comparison)
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_B.default"
            hf_lora_B_in = get_hf_tensor(hf_tensor_name, input_comparison)
            torch.testing.assert_close(hf_lora_A_out, hf_lora_B_in, rtol=1.3e-6, atol=1e-5)
            ff_tensor_name = f"layers.{i}.layers.{i}.mlp.down_proj.lora"
            ff_lora_A_out = get_ff_tensor(ff_tensor_name, output_comparison, hf_lora_A_out.shape, tp_type=TPType.TO_REDUCE)
            compare(hf_lora_A_out, ff_lora_A_out, label=f"LoRA_A {i} output")

            # LoRA_B
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_B.default"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output", ff_tensor_type="output", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison) * self.lora_scaling_factor
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_down_proj_out.shape, tp_type=TPType.TO_REDUCE)
            compare(hf_down_proj_out, ff_tensor, label=f"W2_out + scaling*LoRA_B_out {i}")
            compare(hf_tensor, ff_tensor, additional_ff_tensor=ff_down_proj_out, label=f"LoRA_B {i} output")
        
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

    def check_bwd_pass(self, step_idx=0):
        if not self.num_tokens or not self.ff_batch_size:
            raise ValueError("Number of tokens and batch size must be set before running backward pass check")
        hf_bwd_folder = os.path.join(hf_path, "bwd", f"step_{step_idx}")
        ff_bwd_folder = os.path.join(ff_path, "bwd", f"step_{step_idx}", "shard_0")
        
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
                # f_version = f_version.replace(".q_proj", "").replace(".k_proj", "").replace(".v_proj", "").replace(".o_proj", "")
                # lora in HuggingFace is split into A and B operators, in FF we use a single operator.
                f_version = f_version.replace("lora_A", "lora").replace("lora_B", "lora")
            return f_version
        
        def get_hf_tensor(hf_tensor_name, tensor_comparison_idx):
            hf_tensor_filename = f"{hf_tensor_name}.{tensor_comparison_idx.hf_tensor_type}_{tensor_comparison_idx.hf_tensor_idx}"
            hf_tensor_path = os.path.join(hf_bwd_folder, hf_tensor_filename)

            if not os.path.isfile(hf_tensor_path):
                raise FileNotFoundError(f"File '{hf_tensor_path}' not found")
            print("loading hf tensor: ", hf_tensor_filename)
            hf_tensor = torch.load(hf_tensor_path, map_location='cpu')
            return hf_tensor
        
        def get_ff_tensor(ff_tensor_name, tensor_comparison_idx, hf_shape, tp_type=TPType.REPLICATE, pre=False, shard_axis=0):
            ff_tensor_suffix = f".{tensor_comparison_idx.ff_tensor_type}" if len(tensor_comparison_idx.ff_tensor_type) > 0 else ""
            ff_tensor_idx_suffix = f"_{tensor_comparison_idx.ff_tensor_idx}" if tensor_comparison_idx.ff_tensor_idx is not None else ""
            ff_tensor_filename = f"{ff_tensor_name}{ff_tensor_suffix}{ff_tensor_idx_suffix}"
            
            ff_tensor_path = os.path.join(ff_bwd_folder, ff_tensor_filename)
            if pre:
                ff_tensor_path = ff_tensor_path.replace(f"step_{step_idx}", f"step_{step_idx}_pre")
            if not os.path.isfile(ff_tensor_path):
                raise FileNotFoundError(f"File '{ff_tensor_path}' not found")
            print("loading ff tensor: ", ff_tensor_filename)

            ff_shape = list(hf_shape)[::-1]
            if tp_type == TPType.PARTITION:
                ff_shape[shard_axis] //= self.tp_degree

            # exception: intermediate attention tensors
            intermediate_attention_tensor = (
                "self_attn" in ff_tensor_name and 
                not (
                    ff_tensor_name.endswith(".self_attn") and
                    (
                        tensor_comparison_idx.ff_tensor_type == "output_gradient" or
                        tensor_comparison_idx.ff_tensor_type == "input_gradient"
                    )
                ) and
                not ff_tensor_name.endswith(".self_attn.qkv_proj")
            )
            print(ff_tensor_filename + (" is not truncated" if intermediate_attention_tensor else " is truncated"))
            if not intermediate_attention_tensor:
                ff_shape = replace_value(ff_shape, self.num_tokens, self.ff_batch_size)
            
            ff_tensors = [load_ff_tensor(ff_tensor_path.replace("shard_0", f"shard_{tp_idx}"), ff_shape) for tp_idx in range(self.tp_degree)]
            if self.tp_degree > 1:
                # if replicate, check that they are identical
                if tp_type == TPType.REPLICATE:
                    assert(are_np_arrays_identical(ff_tensors))
                    ff_tensor = ff_tensors[0]
                # if partition, concatenate along the partition dimension
                elif tp_type == TPType.PARTITION:
                    ff_tensor = np.concatenate(ff_tensors, axis=shard_axis)
                # if to_reduce, sum along the partition dimension
                elif tp_type == TPType.TO_REDUCE:
                    ff_tensor = np.sum(ff_tensors, axis=shard_axis)
            else:
                ff_tensor = ff_tensors[0]
            ff_tensor = torch.from_numpy(ff_tensor)
            if not intermediate_attention_tensor:
                ff_tensor = truncate_dimension(ff_tensor, self.ff_batch_size, self.num_tokens)
            return ff_tensor

        def compare(hf_tensor, ff_tensor, label="", additional_ff_tensor=None, tolerance=1e-3):
            ff_tensor = ff_tensor.to(hf_tensor.dtype)
            hf_tensor = hf_tensor.T
            if additional_ff_tensor is not None:
                additional_ff_tensor = additional_ff_tensor.to(hf_tensor.dtype)
                ff_tensor = ff_tensor - additional_ff_tensor
            try:
                # torch.testing.assert_close(hf_tensor, ff_tensor, rtol=rtol, atol=tolerance)
                if not np.allclose(hf_tensor.numpy(), ff_tensor.numpy(), atol=tolerance):
                    mismatches = np.where(~np.isclose(hf_tensor, ff_tensor, atol=tolerance))[0]
                    print(f"Pct mismatch {label}: {100.0*(np.prod(mismatches.shape) / ff_tensor.numel()):.3f}%")
                    assert(np.prod(mismatches.shape) <= .06 * ff_tensor.numel())
            except Exception as e:
                print(f"Error in comparison {label}:\n{e}\n")
                print("HF tensor:")
                print(hf_tensor.squeeze())
                print(hf_tensor.shape)
                print("FF tensor:")
                print(ff_tensor.squeeze())
                print(ff_tensor.shape)
                raise e
        
        print(f"-- BWD pass {step_idx}--")
        
        # LM head
        hf_tensor_name = "lm_head"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
        input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
        compare(hf_tensor, ff_tensor, label="LM head gradient output")
        hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, TPType.TO_REDUCE)
        compare(hf_tensor, ff_tensor, label="LM head gradient input")

        # Norm
        hf_tensor_name = "norm"
        ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
        output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
        input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
        hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
        compare(hf_tensor, ff_tensor, label="Norm gradient output")
        hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
        ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape)
        compare(hf_tensor, ff_tensor, label="Norm gradient input")

        # Transformers blocks
        for i in range(self.num_layers-1, -1, -1):
            # W2 (down_proj) output
            hf_tensor_name = f"layers.{i}.mlp.down_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
            compare(hf_tensor, ff_tensor, label=f"W2 {i} gradient output")

            # LoRA_B
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_B.default"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE) * self.lora_scaling_factor
            compare(hf_tensor, ff_tensor, label=f"LoRA_B {i} gradient output")

            # LoRA_A
            hf_tensor_name = f"layers.{i}.mlp.down_proj.lora_A.default"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"LoRA_A {i} gradient input")

            # W2 (down_proj) input
            hf_tensor_name = f"layers.{i}.mlp.down_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W2 {i} gradient input")
            
            # W2 input (HF) and SigmoidSiluMulti output (FF)
            hf_w2_input = hf_tensor.clone()
            ff_tensor_name = f"layers.{i}.SigmoidSiluMulti"
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_w2_input, ff_tensor, label=f"HF W2 {i} output and FF SSM output")

            # W1 (gate_proj) output
            hf_tensor_name = f"layers.{i}.mlp.gate_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W1 {i} gradient output")
            # W1 (gate_proj) input
            # HF W1 in = FF W1 in - HF W1 in (pre)
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            ff_tensor_pre = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE, pre=True)
            compare(hf_tensor, ff_tensor, additional_ff_tensor=ff_tensor_pre, label=f"W1 {i} gradient input")

            # W3 (up_proj) output
            hf_tensor_name = f"layers.{i}.mlp.up_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"W3 {i} gradient output")
            # W3 (up_proj) input
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            compare(hf_tensor, ff_tensor, label=f"W3 {i} gradient input")

            # Attn O-proj
            hf_tensor_name = f"layers.{i}.self_attn.o_proj"
            ff_tensor_name = f"layers.{i}.layers.{i}.self_attn.o_proj"
            # ff_tensor_name = f"layers.{i}.layers.{i}.self_attn"
            output_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            # hf_tensor = get_hf_tensor(hf_tensor_name, output_comparison)
            # ff_tensor = get_ff_tensor(ff_tensor_name, output_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            # compare(hf_tensor, ff_tensor, label=f"Attn O-proj {i} gradient output")
            ff_tensor_name = f"layers.{i}.layers.{i}.self_attn.o_proj"
            input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.PARTITION)
            compare(hf_tensor, ff_tensor, label=f"Attn O-proj {i} gradient input")

            # V-proj grads
            # FF shape: [num_tokens, qProjSize*num_heads]
            hf_tensor_name = f"layers.{i}.self_attn.v_proj"
            ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
            mixed_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, mixed_comparison)
            hf_tensor = hf_tensor.squeeze().T
            ff_tensor = get_ff_tensor(ff_tensor_name, mixed_comparison, hf_tensor.shape, tp_type=TPType.PARTITION, shard_axis=1)
            compare(hf_tensor, ff_tensor, label=f"V-proj {i} gradient input")

            # K-proj grads
            # FF shape: (num_tokens, qProjSize, num_heads)
            hf_tensor_name = f"layers.{i}.self_attn.k_proj"
            ff_tensor_name = f"layers.{i}.layers.{i}.self_attn"
            k_proj_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="devkproj", hf_tensor_idx=0, ff_tensor_idx=None)
            hf_tensor = get_hf_tensor(hf_tensor_name, k_proj_comparison)
            hf_tensor = hf_tensor.squeeze().view(self.num_tokens, self.num_attention_heads, self.projsize).transpose(1, 2).contiguous()
            hf_tensor = hf_tensor.T
            ff_tensor = get_ff_tensor(ff_tensor_name, k_proj_comparison, hf_tensor.shape, tp_type=TPType.PARTITION, shard_axis=2)
            compare(hf_tensor, ff_tensor, label=f"K-proj {i} gradient input")
            
            # Q-proj grads
            # FF shape (devQKVPRojArray): (num_tokens, qProjSize, num_heads, 3)
            # Q-proj out grad: devQKVPRojArray[:,:,:,0]
            hf_tensor_name = f"layers.{i}.self_attn.q_proj"
            ff_tensor_name = f"layers.{i}.layers.{i}.self_attn.devQKVPRojArray"
            q_proj_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="", hf_tensor_idx=0, ff_tensor_idx=None)
            hf_tensor = get_hf_tensor(hf_tensor_name, q_proj_comparison)
            hf_tensor = hf_tensor.view(self.num_tokens, self.num_attention_heads, self.projsize).transpose(1, 2).contiguous().T
            augmented_hf_tensor_shape = torch.Size([3]+list(hf_tensor.size()))
            ff_tensor = get_ff_tensor(ff_tensor_name, q_proj_comparison, augmented_hf_tensor_shape, tp_type=TPType.PARTITION, shard_axis=2)[:,:,:,0]
            compare(hf_tensor, ff_tensor, label=f"Q-proj {i} gradient input")
            
            # FF Attn input with HF layernorm out
            hf_tensor_name = f"layers.{i}.input_layernorm"
            ff_tensor_name = f"layers.{i}.layers.{i}.self_attn.qkv_proj"
            input_comparison = TensorComparisonIdxs(hf_tensor_type="output_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
            hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
            ff_tensor = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.TO_REDUCE)
            compare(hf_tensor, ff_tensor, label=f"Attn input {i} gradient input")

            if i > 0:
                # FF attn input with FF layernorm out 1
                attn_input = ff_tensor.clone()
                ff_tensor_name = f"layers.{i}.layers.{i}.input_layernorm"
                _output_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="output_gradient", hf_tensor_idx=0, ff_tensor_idx=1)
                input_layernorm_out1 = get_ff_tensor(ff_tensor_name, _output_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
                torch.testing.assert_close(attn_input, input_layernorm_out1, rtol=1.3e-6, atol=1e-5)

                # Input layernorm
                
                hf_tensor_name = f"layers.{i}.input_layernorm"
                ff_tensor_name = convert_hf_filename_to_ff(hf_tensor_name)
                input_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=0)
                ff_in1_comparison = TensorComparisonIdxs(hf_tensor_type="input_gradient", ff_tensor_type="input_gradient", hf_tensor_idx=0, ff_tensor_idx=1)
                input_layernorm0 = get_ff_tensor(ff_tensor_name, input_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
                input_layernorm1 = get_ff_tensor(ff_tensor_name, ff_in1_comparison, hf_tensor.shape, tp_type=TPType.REPLICATE)
                torch.testing.assert_close(input_layernorm0, input_layernorm1, rtol=1.3e-6, atol=1e-5)
                hf_tensor = get_hf_tensor(hf_tensor_name, input_comparison)
                # if i > 1:
                #     compare(hf_tensor, input_layernorm1, label=f"Input layernorm {i} gradient input")

    def check_step(self, step_idx=0, learning_rate=0.001):
        hf_weight_folder = os.path.join(hf_path, "weights", f"step_{step_idx}")
        ff_weight_folder = os.path.join(ff_path, "weights", f"step_{step_idx}", "shard_0")
        def convert_hf_filename_to_ff(hf_filename):
            assert hf_filename.startswith("layers.")
            layernum = hf_filename.split("layers.")[1].split(".")[0]
            f_version = f"layers.{layernum}."
            f_version += hf_filename.replace(".base_layer", "").replace(".default", "")
            # lora in HuggingFace is split into A and B operators, in FF we use a single operator.
            f_version = f_version.replace("lora_A", "lora.weight_A").replace("lora_B", "lora.weight_B")
            return f_version
        def get_hf_tensor(hf_tensor_name):
            hf_tensor_path = os.path.join(hf_weight_folder, hf_tensor_name)

            if not os.path.isfile(hf_tensor_path):
                raise FileNotFoundError(f"File '{hf_tensor_path}' not found")
            hf_tensor = torch.load(hf_tensor_path, map_location='cpu')
            return hf_tensor
        def get_ff_tensor(ff_tensor_name, hf_shape, tp_type=TPType.REPLICATE, pre=False):
            ff_tensor_path = os.path.join(ff_weight_folder, ff_tensor_name)
            if pre:
                ff_tensor_path = ff_tensor_path.replace(f"step_{step_idx}", f"step_{step_idx}_pre")
            if not os.path.isfile(ff_tensor_path):
                raise FileNotFoundError(f"File '{ff_tensor_path}' not found")

            ff_shape = list(hf_shape)[::-1]
            if tp_type == TPType.PARTITION:
                ff_shape[0] //= self.tp_degree
            
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
            return ff_tensor
        def compare(hf_tensor, ff_tensor, label="", tolerance=1e-4):
            ff_tensor = ff_tensor.to(hf_tensor.dtype)
            hf_tensor = hf_tensor.T
            try:
                # torch.testing.assert_close(hf_tensor, ff_tensor, rtol=rtol, atol=tolerance)
                if not np.allclose(hf_tensor.numpy(), ff_tensor.numpy(), atol=tolerance):
                    mismatches = np.where(~np.isclose(hf_tensor, ff_tensor, atol=tolerance))[0]
                    print(f"Pct mismatch {label}: {100.0*(np.prod(mismatches.shape) / ff_tensor.numel()):.3f}%")
                    assert(np.prod(mismatches.shape) <= .05 * ff_tensor.numel())
            except Exception as e:
                print(f"Error in comparison {label}:\n{e}\n")
                print("HF tensor:")
                print(hf_tensor.squeeze())
                print("FF tensor:")
                print(ff_tensor.squeeze())
                raise e
        print(f"-- optimizer pass {step_idx}--")
        
        for i in range(self.num_layers-1, -1, -1):
            # LoRA_B gradient
            hf_gradient_name = f"layers.{i}.mlp.down_proj.lora_B.default.gradient"
            hf_gradient = get_hf_tensor(hf_gradient_name)
            hf_original_weight_name = f"layers.{i}.mlp.down_proj.lora_B.default.weight_original"
            hf_original_weight = get_hf_tensor(hf_original_weight_name)
            hf_finetuned_weight_name = f"layers.{i}.mlp.down_proj.lora_B.default.weight_finetuned"
            hf_finetuned_weight = get_hf_tensor(hf_finetuned_weight_name)
            torch.testing.assert_close(hf_gradient, (hf_original_weight-hf_finetuned_weight)/learning_rate, rtol=1.3e-6, atol=1e-5)
            ff_gradient_name = convert_hf_filename_to_ff(hf_gradient_name)
            ff_gradient = get_ff_tensor(ff_gradient_name, hf_gradient.shape, tp_type=TPType.REPLICATE)
            compare(hf_gradient, ff_gradient, label=f"LoRA_B {i} gradient")
            # ff_out_gradient_name = f"layers.{i}.layers.{i}.mlp.down_proj.lora.output_gradient_0"
            # ff_fwd_folder = os.path.join(ff_path, "fwd", f"step_{step_idx}", "shard_0")
            # ff_bwd_folder = os.path.join(ff_path, "bwd", f"step_{step_idx}", "shard_0")
            # ff_out_gradient = load_ff_tensor(os.path.join(ff_bwd_folder, ff_out_gradient_name), [self.hidden_size, 128])[:,:self.num_tokens]
            # ff_out_gradient = torch.from_numpy(ff_out_gradient)
            # print("Output gradient shape: ", ff_out_gradient.shape)
            # ff_low_rank_activation = f"layers.{i}.layers.{i}.mlp.down_proj.lora.low_rank_activation"
            # ff_low_rank_activation = load_ff_tensor(os.path.join(ff_fwd_folder, ff_low_rank_activation), [16, 128])[:,:self.num_tokens]
            # ff_low_rank_activation = torch.from_numpy(ff_low_rank_activation)
            # print("Low rank activation shape: ", ff_low_rank_activation.shape)
            # simulated_weight_grad = ff_low_rank_activation @ ff_out_gradient.T
            # print("Simulated weight grad shape: ", simulated_weight_grad.shape)
            # print(simulated_weight_grad)
            # print(ff_gradient)
            # compare(hf_gradient, simulated_weight_grad, label=f"LoRA_B {i} simulated gradient")
            

            # LoRA_A gradient
            hf_gradient_name = f"layers.{i}.mlp.down_proj.lora_A.default.gradient"
            hf_gradient = get_hf_tensor(hf_gradient_name)
            ff_gradient_name = convert_hf_filename_to_ff(hf_gradient_name)
            hf_original_weight_name = f"layers.{i}.mlp.down_proj.lora_A.default.weight_original"
            hf_original_weight = get_hf_tensor(hf_original_weight_name)
            hf_finetuned_weight_name = f"layers.{i}.mlp.down_proj.lora_A.default.weight_finetuned"
            hf_finetuned_weight = get_hf_tensor(hf_finetuned_weight_name)
            torch.testing.assert_close(hf_gradient, (hf_original_weight-hf_finetuned_weight)/learning_rate, rtol=1.3e-6, atol=1e-5)
            ff_gradient_name = convert_hf_filename_to_ff(hf_gradient_name)
            ff_gradient = get_ff_tensor(ff_gradient_name, hf_gradient.shape, tp_type=TPType.PARTITION)
            compare(hf_gradient, ff_gradient, label=f"LoRA_A {i} gradient")

parser = argparse.ArgumentParser(description='Argument Parser Example') 
# Adding arguments
parser.add_argument('-m', '--model-name', type=str, default="goliaro/llama-160m-lora", help='Name of the model')
parser.add_argument('-n', '--num-steps', type=int, default=1, help='Number of finetuning steps')
parser.add_argument('-tp', '--tensor-parallelism-degree', type=int, default=1, help='The tensor parallelism degree used when running FlexFlow')
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='The learning rate used at finetuning time')

# Parse the arguments from command line
args = parser.parse_args()

if __name__ == "__main__":
    llama_alignment = LllamaAlignmentTest(args.model_name, tp_degree=args.tensor_parallelism_degree)
    # llama_alignment.check_weights_alignment()
    for i in range(args.num_steps):
        llama_alignment.check_fwd_pass(i)
        llama_alignment.check_bwd_pass(i)
        llama_alignment.check_step(i, args.learning_rate)
