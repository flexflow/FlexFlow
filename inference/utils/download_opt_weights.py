#!/usr/bin/env python

import os
import requests
import argparse
from transformers import AutoModelForCausalLM

# You can pass the --use-full-precision flag to use the full-precision weight. By default, we use half precision.
parser = argparse.ArgumentParser()
parser.add_argument("--use-full-precision", action="store_true", help="Use full precision")
args = parser.parse_args()
if not args.use_full_precision:
    import torch
    torch.set_default_tensor_type(torch.HalfTensor)

# Change working dir to folder storing this script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def convert_hf_model(model, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for name, params in model.named_parameters():
        name = (
            name.replace(".", "_")
            .replace("decoder_", "")
            .replace("model_", "")
            .replace("self_attn", "attention")
            .replace("q_proj", "wq")
            .replace("k_proj", "wk")
            .replace("v_proj", "wv")
            .replace("out_proj", "wo")
        )
        params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")

# Download and convert big model weights
model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")
dst_folder="../weights/opt_6B_weights" if args.use_full_precision else "../weights/opt_6B_weights_half"
convert_hf_model(model, dst_folder)

# Download and convert small model weights
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
dst_folder="../weights/opt_125M_weights" if args.use_full_precision else "../weights/opt_125M_weights_half"
convert_hf_model(model, dst_folder)

# Download tokenizer files
os.makedirs("../tokenizer", exist_ok=True)
tokenizer_filepath = '../tokenizer/gpt2-vocab.json'
url = 'https://raw.githubusercontent.com/facebookresearch/metaseq/main/projects/OPT/assets/gpt2-vocab.json'
r = requests.get(url)
open(tokenizer_filepath , 'wb').write(r.content)
tokenizer_filepath = '../tokenizer/gpt2-merges.txt'
url = 'https://raw.githubusercontent.com/facebookresearch/metaseq/main/projects/OPT/assets/gpt2-merges.txt'
r = requests.get(url)
open(tokenizer_filepath , 'wb').write(r.content)
