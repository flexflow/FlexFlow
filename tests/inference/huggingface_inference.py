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
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "peft"))
from hf_utils import *

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
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)
    
    # Run huggingface model
    cuda_availble = torch.cuda.is_available()
    device = "cuda" if args.gpu and cuda_availble else "cpu"
    # Get Model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    # Get Tokenizer
    hf_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model_name)
    generation_config.do_sample = args.do_sample
    if not args.do_sample:
        generation_config.num_beams=1
        generation_config.temperature = None
        generation_config.top_p = None
    ################# debugging #################
    if args.inference_debugging:
        # Print model and configs
        print(hf_config)
        print(model)
        make_debug_dirs()
        register_inference_hooks(model)
        # Save weights
        save_model_weights(model, target_modules=["lora", "lm_head", "final_layer_norm", "self_attn_layer_norm", "out_proj", "fc1", "fc2"])

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
