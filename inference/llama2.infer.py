# import argparse
# import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


# if torch.cuda.is_available:
#    device = torch.device('cuda')
# else:
#    device = torch.device('cpu')

torch.set_default_tensor_type(torch.HalfTensor)
# torch.set_default_tensor_type(torch.cuda.HalfTensor)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

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
        params.detach().cpu().numpy().tofile("/home/ubuntu/FlexFlow/inference/weights/llama2_7B_hf_weights_half/" + name)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf", use_fast=True)

# prompt = "Give three tips for staying healthy."

# tokens = tokenizer(
#                 prompt, return_tensors="pt", add_special_tokens=True
#             )

# generated = model.generate(tokens["input_ids"], max_length=128)
# out = tokenizer.decode(generated[0])

# print(out)