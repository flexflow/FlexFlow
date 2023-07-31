import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)



prompt = "Give three tips for staying healthy."
token = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )
print(token["input_ids"])
generated = model.generate(token["input_ids"], max_length=20)
out = tokenizer.decode(generated[0])
print(out)
