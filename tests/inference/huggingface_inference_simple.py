from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
do_sample = False
max_length = 128
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto",)
hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(model_name)
print(generation_config.do_sample)
generation_config.do_sample = do_sample
generation_config.num_beams=1
generation_config.temperature = None
generation_config.top_p = None


def run_text_completion():
    prompt = "Help me plan a 1-week trip to Dubai"
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    generated = model.generate(
        batch["input_ids"],
        max_new_tokens=max_length,
        generation_config=generation_config,
    )
    out = tokenizer.decode(generated[0])
    print(out)

def run_chat_completion():
    messages=[
        {"role": "system", "content": "You are a helpful an honest programming assistant."},
        {"role": "user", "content": "Is Rust better than Python?"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(tokenized_chat, return_tensors="pt")

    generated = model.generate(
        batch["input_ids"],
        max_new_tokens=max_length,
        generation_config=generation_config,
    )
    out = tokenizer.decode(generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    prompt_length = len(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    all_text = out[prompt_length:]
    print(all_text)
run_chat_completion()