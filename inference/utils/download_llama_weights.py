#!/usr/bin/env python

import os
import requests
import argparse
from transformers import AutoModelForCausalLM

# You can pass the --use-full-precision flag to use the full-precision weight. By default, we use half precision.
# and pass "--use_13B", "--use_30B", and "--use_65B" to use the corresponding "llama-13B/30B/65B" model weights
parser = argparse.ArgumentParser()
parser.add_argument("--use-full-precision", action="store_true", help="Use full precision")
parser.add_argument("--use_13B", action="store_true", help="choose to use llama-13B")
parser.add_argument("--use_30B", action="store_true", help="choose to use llama-30B")
parser.add_argument("--use_65B", action="store_true", help="choose to use llama-65B")
args = parser.parse_args()
if not args.use_full_precision:
    import torch
    torch.set_default_tensor_type(torch.HalfTensor)

# Change working dir to folder storing this script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print(dname)
os.chdir(dname)

def convert_hf_model(model, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for name, params in model.named_parameters():
        name = (
            name.replace(".", "_")
            .replace("self_attn", "attention")
            .replace("q_proj", "wq")
            .replace("k_proj", "wk")
            .replace("v_proj", "wv")
            .replace("o_proj", "wo")
            .replace("mlp", "feed_forward")
            .replace("gate_proj", "w1")
            .replace("down_proj", "w2")
            .replace("up_proj", "w3")
            .replace("input_layernorm", "attention_norm")
            .replace("post_attention_layernorm", "ffn_norm")
            .replace("embed_tokens", "tok_embeddings")
            .replace("lm_head", "output")
            .replace("model_", "")
        )
        params.detach().cpu().numpy().tofile(f"{dst_folder}/{name}")

# Download and convert big model weights
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
dst_folder="../weights/llama_7B_weights" if args.use_full_precision else "../weights/llama_7B_weights_half"
convert_hf_model(model, dst_folder)

# Download and convert model weights only for hf
if args.use_13B:
    model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-13b-hf")
    dst_folder="../weights/llama_13B_weights_half"
    convert_hf_model(model, dst_folder)

if args.use_30B:
    model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-30b-hf")
    dst_folder="../weights/llama_30B_weights_half"
    convert_hf_model(model, dst_folder)

if args.use_65B:
    model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-65b-hf")
    dst_folder="../weights/llama_65B_weights_half"
    convert_hf_model(model, dst_folder)

# Download and convert small model weights
model = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")
dst_folder="../weights/llama_160M_weights" if args.use_full_precision else "../weights/llama_160M_weights_half"
convert_hf_model(model, dst_folder)

# Download tokenizer
os.makedirs("../tokenizer", exist_ok=True)
tokenizer_filepath = '../tokenizer/tokenizer.model'
url = 'https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer.model'
r = requests.get(url)
open(tokenizer_filepath , 'wb').write(r.content)
