from transformers import AutoModelForCausalLM
import os
import torch

torch.set_default_tensor_type(torch.FloatTensor)
dst_folder = "/home/ubuntu/FlexFlow/inference/weights/JackFram/llama-160m/full-precision"
model = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")
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