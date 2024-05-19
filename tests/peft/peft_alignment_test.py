import numpy as np
import os, torch
from alignment.align_test_utils import *

def convert_hf_filename_to_ff_filename(f, num_layers=12):
    if f.endswith(".lm_head.weight"):
        f_version = f"fwd_step_0_layers_{num_layers-1}_lm_head_shard_0_weight_0"
    elif f == "norm.weight":
        f_version = f"fwd_step_0_layers_{num_layers-1}_norm_shard_0_weight_0"
    else:
        f_version = "fwd_step_0_"
        if f.startswith("layers."):
            layernum = f.split("layers.")[1].split(".")[0]
            f_version += f"layers_{layernum}_"
        f_version += f.split(".weight")[0].replace(".base_layer", "").replace(".default", "")
        weight_index="0"
        if "lora_A" in f_version:
            weight_index="A"
        elif "lora_B" in f_version:
            weight_index="B"
        f_version = f_version.replace("lora_A", "lora").replace("lora_B", "lora")
        f_version += f"_shard_0_weight_{weight_index}"
    return f_version

def check_weights_alignment():
    print("-- Weights alignment --")
    files_list = os.listdir(hf_path)
    num_layers=12
    for f in sorted(files_list):
        if f.endswith(".weight"):
            if "self_attn" in f:
                continue
            f_version = convert_hf_filename_to_ff_filename(f, num_layers=num_layers)
            # print(f, f_version)
            hf_w_path = os.path.join(hf_path, f)
            ff_w_path = os.path.join(ff_path, f_version)
            assert(os.path.isfile(hf_w_path))
            assert(os.path.isfile(ff_w_path))
            # print("\t", os.path.isfile(hf_w_path), os.path.isfile(ff_w_path))
            # print("\t", ff_w_path)

            # check equivalence
            compare_tensors(hf_w_path, ff_w_path, tolerance=1e-5)

def check_fwd_pass(tot_num_layers = 12):
    print("-- FWD pass --")
    # Transfomer head
    hf_embed_input= f"{hf_path}/fwd_step_0_embed_tokens.input_0"
    ff_embed_input = f"{ff_path}/fwd_step_0_layers_0_embed_tokens_shard_0_input_0"
    compare_tensors(hf_embed_input, ff_embed_input)
    hf_embed_output = f"{hf_path}/fwd_step_0_embed_tokens.output_0"
    ff_embed_output = f"{ff_path}/fwd_step_0_layers_0_embed_tokens_shard_0_output_0"
    compare_tensors(hf_embed_output, ff_embed_output)

    # Transformers blocks
    for i in range(tot_num_layers):
        hf_input_ln_in = f"{hf_path}/fwd_step_0_layers.{i}.input_layernorm.input_0"
        ff_input_ln_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_input_0"
        if i > 0:
            ff_input_ln_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_0"
        compare_tensors(hf_input_ln_in, ff_input_ln_in, tolerance=1e-5)
        hf_input_ln_out = f"{hf_path}/fwd_step_0_layers.{i}.input_layernorm.output_0"
        ff_input_ln_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_0"
        if i > 0:
            ff_input_ln_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.input_layernorm_shard_0_output_1"
        compare_tensors(hf_input_ln_out, ff_input_ln_out, tolerance=1e-5)
        hf_attn_out = f"{hf_path}/fwd_step_0_layers.{i}.self_attn.o_proj.output_0"
        ff_attn_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.self_attn_shard_0_output_0"
        compare_tensors(hf_attn_out, ff_attn_out, tolerance=1e-5)
        hf_ffn_norm_out = f"{hf_path}/fwd_step_0_layers.{i}.post_attention_layernorm.output_0"
        ff_ffn_norm_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.post_attention_layernorm_shard_0_output_1"
        compare_tensors(hf_ffn_norm_out, ff_ffn_norm_out, tolerance=1e-5)
        # w1
        hf_gate_proj_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.gate_proj.output_0"
        ff_gate_proj_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.gate_proj_shard_0_output_0"
        compare_tensors(hf_gate_proj_out, ff_gate_proj_out, tolerance=1e-5)
        # w3
        hf_up_proj_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.up_proj.output_0" 
        ff_up_proj_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.up_proj_shard_0_output_0"
        compare_tensors(hf_up_proj_out, ff_up_proj_out, tolerance=1e-5)
        # w2
        hf_down_proj_in = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.input_0"
        hf_down_proj_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.output_0"
        ff_down_proj_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_input_0"
        ff_down_proj_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj_shard_0_output_0"
        compare_tensors(hf_down_proj_in, ff_down_proj_in)
        # compare_tensors(hf_down_proj_out, ff_down_proj_out)
        # LORA input
        hf_lora_A_in = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_A.default.input_0"
        ff_lora_A_in = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_input_0"
        compare_hf_tensors(hf_down_proj_in, hf_lora_A_in)
        compare_tensors(hf_lora_A_in, ff_lora_A_in)
        # LORA weights
        hf_lora_A_weight_fp = f"{hf_path}/layers.{i}.mlp.down_proj.lora_A.default.weight"
        ff_lora_A_weight_fp = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_A"
        compare_tensors(hf_lora_A_weight_fp, ff_lora_A_weight_fp)
        hf_lora_B_weight_fp = f"{hf_path}/layers.{i}.mlp.down_proj.lora_B.default.weight"
        ff_lora_B_weight_fp = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_weight_B"
        compare_tensors(hf_lora_B_weight_fp, ff_lora_B_weight_fp)
        # LORA intermediate hf
        hf_lora_A_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_A.default.output_0"
        hf_lora_B_in = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_B.default.input_0"
        compare_hf_tensors(hf_lora_A_out, hf_lora_B_in)
        # LORA output
        hf_lora_out = f"{hf_path}/fwd_step_0_layers.{i}.mlp.down_proj.lora_B.default.output_0"
        ff_lora_out = f"{ff_path}/fwd_step_0_layers_{i}_layers.{i}.mlp.down_proj.lora_shard_0_output_0"
        # compare_tensors(hf_lora_out, ff_lora_out)
        # compare_flexflow_tensors(ff_down_proj_out, ff_lora_out)
        # compare_tensors(hf_down_proj_out, ff_lora_out)
        compare_tensors_difference(hf_lora_out, ff_lora_out, ff_down_proj_out)
        

    # After last layer only
    hf_norm_out = f"{hf_path}/fwd_step_0_norm.output_0"
    ff_norm_out = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_output_1"
    compare_tensors(hf_norm_out, ff_norm_out, tolerance=1e-5)
    hf_lm_head_out = f"{hf_path}/fwd_step_0_base_model.model.lm_head.output_0"
    ff_lm_head_out = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_lm_head_shard_0_output_0"
    compare_tensors(hf_lm_head_out, ff_lm_head_out, tolerance=1e-5)

def check_bwd_pass(tot_num_layers = 12):
    # ff_BWD_softmax_in = f"{ff_path}/model_0_bwd-step_0_layer-num_100_layer-name_Softmax_shard-id_0_input_0"
    print("-- LM head --")
    hf_BWD_lm_head_out = f"{hf_path}/bwd_step_0_base_model.model.lm_head.go_0"
    ff_BWD_lm_head_out = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_lm_head_shard_0_output_0"
    compare_tensors(hf_BWD_lm_head_out, ff_BWD_lm_head_out, tolerance=1e-5)
    # compare weights
    # hf_lm_head_weight = f"{hf_path}/base_model.model.lm_head.weight"
    # ff_lm_head_weight = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_output_shard_0_weight_0"
    # compare_tensors(hf_lm_head_weight, ff_lm_head_weight, tolerance=1e-5)
    hf_BWD_lm_head_in = f"{hf_path}/bwd_step_0_base_model.model.lm_head.gi_0"
    ff_BWD_lm_head_in = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_lm_head_shard_0_input_0"
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
    hf_BWD_norm_out = f"{hf_path}/bwd_step_0_norm.go_0"
    ff_BWD_norm_out = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_output_0"
    compare_hf_tensors(hf_BWD_lm_head_in, hf_BWD_norm_out)
    compare_tensors(hf_BWD_norm_out, ff_BWD_norm_out)
    ff_BWD_norm_weight = f"{ff_path}/fwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_weight_0"
    hf_FWD_norm_weight = f"{hf_path}/norm.weight"
    compare_tensors(hf_FWD_norm_weight, ff_BWD_norm_weight, tolerance=1e-5)
    hf_BWD_norm_in = f"{hf_path}/bwd_step_0_norm.gi_0"
    ff_BWD_norm_in = f"{ff_path}/bwd_step_0_layers_{tot_num_layers-1}_norm_shard_0_input_1"
    compare_tensors(hf_BWD_norm_in, ff_BWD_norm_in, tolerance=1e-5)

if __name__ == "__main__":
    check_weights_alignment()
    check_fwd_pass()
    check_bwd_pass()
