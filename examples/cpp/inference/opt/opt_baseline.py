from transformers import OPTConfig, OPTForCausalLM, GPT2Tokenizer

model_id = "facebook/opt-125m"
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = OPTForCausalLM.from_pretrained(model_id)

prompts = [
            "Today is a beautiful day and I want",
        ]

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    print(input_ids)
    generated_ids = model.generate(input_ids, max_length=11)
    generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_ids)
    print(generated_string)

# tensor([[   2, 5625,   16,   10, 2721,  183,    8,   38,  236,    7,  458,   19,
#            47,    5, 2770,  527,    9,  127,   78,  655, 1805]])

##this is a release version, the codebase is in https://github.com/huggingface/transformers/tree/v4.28-release
## our current align base is the main branch, the result will be different
