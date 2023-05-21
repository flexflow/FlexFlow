#!/usr/bin/env python

import os
import requests
from transformers import AutoModelForCausalLM

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
            .replace("embed_tokens_weight", "embed_tokens_weight_lm_head")
        )
        params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")

# Download and convert big model weights
model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")
dst_folder="../weights/opt_6B_weights"
convert_hf_model(model, dst_folder)

# Download and convert small model weights
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
dst_folder="../weights/opt_125M_weights"
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
