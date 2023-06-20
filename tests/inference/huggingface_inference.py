import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

def main():
    # Change working dir to folder storing this script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tokenizer-model-name", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument(
        "--use-full-precision", action="store_true", help="Use full precision"
    )
    parser.add_argument("--gpu", action="store_true", help="Run on GPU")
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
        import torch

        torch.set_default_tensor_type(torch.HalfTensor)

    # Run huggingface model
    device = "cuda" if args.gpu else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    if args.tokenizer_model_name == "JackFram/llama-160m":
        tokenizer = LlamaTokenizer.from_pretrained("JackFram/llama-160m", use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name)
    with open(args.output_file, "w") as f:
        for i, prompt in enumerate(prompt_list):
            batch = tokenizer(
                prompt_list, return_tensors="pt", add_special_tokens=True
            ).to(device)
            generated = model.generate(batch["input_ids"], max_length=args.max_length)
            out = tokenizer.decode(generated[0])
            # Write output to file
            out_str = out if i == (len(prompt_list) - 1) else out + "\n"
            f.write(out_str)


if __name__ == "__main__":
    main()
