import argparse
import json
import os
import shutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    GenerationConfig,
)
######################### debugging helper functions #########################
def pre_forward_hook(module, input):
    assert module.name is not None and module.decoding_step is not None
    name = module.name.replace("model.", "")
    print(
        f"Pre-forward hook activated on module: {name}, decoding step: {module.decoding_step}"
    )
    print("Pre-Input: ", input[0].shape)
    torch.save(
        input, f"./hf_tensors/decoding_step_{module.decoding_step}_{name}.input"
    )
def post_forward_hook(module, input, output):
    assert module.name is not None and module.decoding_step is not None
    name = module.name.replace("model.", "")
    print(
        f"Post-forward Hook activated for module: {name}, decoding step: {module.decoding_step}"
    )
    print("Post-Input/Output: ", input[0].shape, output[0].shape)
    torch.save(
        output, f"./hf_tensors/decoding_step_{module.decoding_step}_{name}.output"
    )
    print("===")
    module.decoding_step += 1
##############################################################################

def main():
    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument(
        "--use-full-precision", action="store_true", help="Use full precision"
    )
    parser.add_argument("--do-sample", action="store_true", help="Use sampling")
    parser.add_argument("--gpu", action="store_true", help="Run on GPU")
    parser.add_argument(
        "--inference-debugging",
        action="store_true",
        help="Print debugging info and save hidden states/weights to file",
    )
    args = parser.parse_args()
    # Check if max-length is greater than 0
    if args.max_length <= 0:
        print("Error: max-length must be greater than 0.")
        return
    # Check if prompt-file exists
    if not os.path.isfile(args.prompt_file):
        print(f"Error: {args.prompt_file} does not exist.")
        return

    # Read prompt-file into a list of strings
    with open(args.prompt_file, "r") as f:
        try:
            prompt_list = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Unable to parse {args.prompt_file} as JSON.")
            return

    # Set default tensor type depending on argument indicating the float type to use
    if not args.use_full_precision:
        torch.set_default_tensor_type(torch.HalfTensor)

    # Run huggingface model
    cuda_availble = torch.cuda.is_available()
    device = "cuda" if args.gpu and cuda_availble else "cpu"
    # Get Model
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    # Get Tokenizer
    hf_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    hf_arch = getattr(hf_config, "architectures")[0]
    if hf_arch == "LLaMAForCausalLM" or hf_arch == "LlamaForCausalLM":
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    generation_config = GenerationConfig.from_pretrained(args.model_name)
    generation_config.do_sample = args.do_sample
    ################# debugging #################
    if args.inference_debugging:
        # Print model and configs
        print(hf_config)
        print(model)
        # Save weights to file
        shutil.rmtree("./hf_tensors")
        # Check that the output folder exists
        os.makedirs("./hf_tensors", exist_ok=True)
        # Save weights
        for name, params in model.named_parameters():
            torch.save(params, f"./hf_tensors/{name}")
            # params.detach().cpu().numpy().tofile(f"./hf_tensors/{name}")
        # Register hooks to save per-op hidden states
        for name, layer in dict(model.named_modules()).items():
            layer.name = name
            layer.decoding_step = 0
            print(f"Adding hooks to layer {layer.name}")
            layer.register_forward_pre_hook(pre_forward_hook)
            layer.register_forward_hook(post_forward_hook)
    ###############################################
    # Generate output
    with open(args.output_file, "w") as f:
        for i, prompt in enumerate(prompt_list):
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
                device
            )
            generated = model.generate(
                batch["input_ids"],
                max_length=args.max_length,
                generation_config=generation_config,
            )
            out = tokenizer.decode(generated[0])
            # Write output to file
            out_str = out if i == (len(prompt_list) - 1) else out + "\n"
            f.write(out_str)


if __name__ == "__main__":
    main()
