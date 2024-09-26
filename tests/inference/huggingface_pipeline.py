import transformers
from transformers import GenerationConfig

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
do_sample = False

generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.do_sample = do_sample
generation_config.num_beams=1
# generation_config.max_length = 128
generation_config.temperature = None
generation_config.top_p = None
print(generation_config)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages=[
        {"role": "system", "content": "You are a helpful an honest programming assistant."},
        {"role": "user", "content": "Is Rust better than Python?"},
    ]
    
# messages="Help me plan a 1-week trip to Dubai"
outputs = pipeline(
    messages,
    max_new_tokens=128,
    generation_config=generation_config,
)
print(outputs[0]["generated_text"][-1]['content'])