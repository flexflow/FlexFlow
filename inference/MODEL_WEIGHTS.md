To convert the weights of a HuggingFace LLM to SpecInfer's weight format, we first load the model and modify the tensor names to match SpecInfer's convention, and then convert these tensors to numpy arrays to store them in binary files.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

for name, params in model.named_parameters():
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
    params.detach().cpu().numpy().tofile('weights/llama_7B_weights/' + name)
```

