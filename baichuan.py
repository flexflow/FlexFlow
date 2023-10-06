import flexflow.serve as ff

# Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
ff.init(
        num_gpus=4,
        memory_per_gpu=14000,
        zero_copy_memory_per_node=30000,
        tensor_parallelism_degree=4,
        pipeline_parallelism_degree=1
    )

# Create the FlexFlow LLM
llm = ff.LLM("baichuan-inc/baichuan-7B")

# Create the sampling configs
generation_config = ff.GenerationConfig(
    do_sample=True, temperature=0.9, topp=0.8, topk=1
)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config)
print(f"llm:{llm}")
# Generation begins!
result = llm.generate("Here are some travel tips for Tokyo:\n")
print("result:",result)