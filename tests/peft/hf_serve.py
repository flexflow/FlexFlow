import argparse
import torch
import os, sys, shutil, json
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    GenerationConfig,
)


def peft_pre_forward_hook(module, input):
    assert module.name is not None and module.decoding_step is not None
    name = module.name.replace("base_model.model.model.", "")
    print(
        f"Pre-forward hook activated on module: {name}, decoding step: {module.decoding_step}"
    )
    print("Pre-Input: ", input[0].shape)
    torch.save(
        input, f"./hf_peft_tensors/decoding_step_{module.decoding_step}_{name}.input"
    )
    # print("===")


def peft_post_forward_hook(module, input, output):
    assert module.name is not None and module.decoding_step is not None
    name = module.name.replace("base_model.model.model.", "")
    print(
        f"Post-forward Hook activated for module: {name}, decoding step: {module.decoding_step}"
    )
    print("Post-Input/Output: ", input[0].shape, output[0].shape)
    torch.save(
        output, f"./hf_peft_tensors/decoding_step_{module.decoding_step}_{name}.output"
    )
    print("===")
    module.decoding_step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft-model-id", type=str, required=True)
    parser.add_argument(
        "--use-full-precision", action="store_true", help="Use full precision"
    )
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--do-sample", action="store_true", help="Use sampling")
    parser.add_argument(
        "--save-peft-tensors",
        action="store_true",
        help="Save PEFT hidden states and weights to file",
    )
    args = parser.parse_args()

    # Check if prompt-file exists
    if not os.path.isfile(args.prompt_file):
        print(f"Error: {args.prompt_file} does not exist.")
        return

    # Get peft model config
    config = PeftConfig.from_pretrained(args.peft_model_id)
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        # load_in_8bit=True,
        torch_dtype=torch.float32 if args.use_full_precision else torch.float16,
        device_map="auto",
    )
    # Load the Lora model
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    print(model)
    
    # Get tokenizer
    hf_config = AutoConfig.from_pretrained(
        config.base_model_name_or_path, trust_remote_code=True
    )
    hf_arch = getattr(hf_config, "architectures")[0]
    if hf_arch == "LLaMAForCausalLM" or hf_arch == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(
            config.base_model_name_or_path,
            use_fast=True,
            torch_dtype=torch.float32 if args.use_full_precision else torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float32 if args.use_full_precision else torch.float16,
        )
    
    # Generation config
    generation_config = GenerationConfig.from_pretrained(config.base_model_name_or_path)
    generation_config.do_sample = args.do_sample

    # Register hooks to save tensors, if needed
    if args.save_peft_tensors:
        # Change working dir to folder storing this script
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        # Create output dir
        shutil.rmtree("./hf_peft_tensors")
        os.makedirs("./hf_peft_tensors", exist_ok=True)
        # Save weights
        for name, params in model.named_parameters():
            if "lora" in name:
                torch.save(params, f"./hf_peft_tensors/{name}")
                # params.detach().cpu().numpy().tofile(f"{weights_path}/{name}")
        # Save hidden states
        for name, layer in dict(model.named_modules()).items():
            if "lora_A.default" in name or "lora_B.default" in name:
                layer.name = name
                layer.decoding_step = 0
                print(f"Adding hooks to layer {layer.name}")
                layer.register_forward_pre_hook(peft_pre_forward_hook)
                layer.register_forward_hook(peft_post_forward_hook)

    # Run inference
    # Read prompt-file into a list of strings
    with open(args.prompt_file, "r") as f:
        try:
            prompt_list = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Unable to parse {args.prompt_file} as JSON.")
            sys.exit(1)
    
    for i, prompt in enumerate(prompt_list):
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(
                **batch, max_new_tokens=args.max_length, generation_config=generation_config
            )
        print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
