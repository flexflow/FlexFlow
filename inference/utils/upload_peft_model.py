#!/usr/bin/env python
import argparse, os
from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel
import torch
import numpy as np
import flexflow.serve as ff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a PEFT model with FlexFlow, process it, and upload it to the Hugging Face Hub."
    )
    parser.add_argument(
        "peft_model_id",
        type=str,
        help="Hugging Face model ID of the PEFT model to upload.",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=os.environ.get(
            "FF_CACHE_PATH", os.path.expanduser("~/.cache/flexflow")
        ),
        help="Path to the FlexFlow cache folder",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to upload the processed PEFT model as a private model on Hugging Face Hub.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure Hugging Face CLI is logged in
    if not HfFolder.get_token():
        raise RuntimeError(
            "Hugging Face token not found. Please login using `huggingface-cli login`."
        )

    lora_config_filepath = os.path.join(
        args.cache_folder,
        "finetuned_models",
        args.peft_model_id,
        "config",
        "ff_config.json",
    )
    peft_config = ff.LoraLinearConfig.from_jsonfile(lora_config_filepath)
    print(peft_config)
    hf_peft_config = peft_config.to_hf_config()
    print(hf_peft_config)
    if peft_config.precision != "fp32" and peft_config.precision != "fp16":
        raise ValueError(f"Unsupported precision: {peft_config.precision}")
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float32 if peft_config.precision == "fp32" else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.peft_model_id, config=hf_peft_config)
    in_dim = model.config.intermediate_size
    out_dim = model.config.hidden_size

    weight_folder = os.path.join(
        args.cache_folder, "finetuned_models", args.peft_model_id, "weights", "shard_0"
    )
    num_shards = 1
    while os.path.exists(weight_folder.replace("shard_0", f"shard_{num_shards}")):
        num_shards += 1
    if not in_dim % num_shards == 0:
        raise ValueError(
            f"Number of shards ({num_shards}) must divide the input dimension ({in_dim})"
        )
    lora_weight_files = os.listdir(weight_folder)
    for lora_file in sorted(lora_weight_files):
        lora_filename = ".weight".join(lora_file.split(".weight")[:-1])
        hf_parameter_name = f"base_model.model.model.{lora_filename}.default.weight"
        if hf_parameter_name not in model.state_dict().keys():
            raise KeyError(f"Parameter {lora_file} not found in HF model.")

        ff_dtype = np.float32 if peft_config.precision == "fp32" else np.float16
        weight_path = os.path.join(weight_folder, lora_file)
        # LoRA_A: [in_dim, rank]
        # LoRA_B: [rank, out_dim]
        if "lora_A" in lora_file:
            weight_data = []
            for shard_id in range(num_shards):
                weight_path_shard = weight_path.replace("shard_0", f"shard_{shard_id}")
                weight_data_shard = np.fromfile(weight_path_shard, dtype=ff_dtype)
                weight_data_shard = weight_data_shard.reshape(
                    (in_dim // num_shards, peft_config.rank), order="F"
                )
                weight_data.append(weight_data_shard)
            weight_data = np.concatenate(weight_data, axis=0).T
        elif "lora_B" in lora_file:
            weight_data = np.fromfile(weight_path, dtype=ff_dtype)
            weight_data = weight_data.reshape((peft_config.rank, out_dim), order="F").T
        weight_tensor = torch.from_numpy(weight_data)

        param = model.state_dict()[hf_parameter_name]

        actual_numel = weight_tensor.numel()
        expected_numel = param.numel()
        if actual_numel != expected_numel:
            raise ValueError(
                f"Parameter {lora_file} has unexpected parameter count: {actual_numel} (actual) != {expected_numel} (expected)"
            )

        if weight_tensor.shape != param.shape:
            raise ValueError(
                f"Parameter {lora_file} has unexpected shape: {weight_tensor.shape} (actual) != {param.shape} (expected)"
            )
        if weight_tensor.dtype != param.dtype:
            raise ValueError(
                f"Parameter {lora_file} has unexpected dtype: {weight_tensor.dtype} (actual) != {param.dtype} (expected)"
            )

        with torch.no_grad():
            param.copy_(weight_tensor)

    model.push_to_hub(f"{args.peft_model_id}2", use_auth_token=True)

    print("Upload process completed.")


if __name__ == "__main__":
    main()
