import flexflow.serve as ff
import json, os
from types import SimpleNamespace
from datasets import load_dataset
import random

configs_dict = {
    "num_gpus": 1,
    "memory_per_gpu": 8192,
    "zero_copy_memory_per_node": 12000,
    "num_cpus": 4,
    "legion_utility_processors": 4,
    "data_parallelism_degree": 1,
    "tensor_parallelism_degree": 1,
    "pipeline_parallelism_degree": 1,
    "offload": False,
    "offload_reserve_space_size": 8 * 1024,  # 8GB
    "use_4bit_quantization": False,
    "use_8bit_quantization": False,
    "enable_peft": True,
    "peft_activation_reserve_space_size": 1024,  # 1GB
    "peft_weight_reserve_space_size": 1024,  # 1GB
    "profiling": False,
    "inference_debugging": False,
    "fusion": False,
    "seed": 42,
}
model_configs = {
    "base_model": "JackFram/llama-160m",
    "inference_peft_model_id": "goliaro/llama-160m-lora",
    "finetuning_peft_model_id": "goliaro/llama-160m-lora",
    "cache_path": "",
    "refresh_cache": False,
    "full_precision": True,
    # relative paths
    "prompt": "inference_dataset.json",
    "finetuning_dataset": "finetuning_dataset.json",
    "output_file": "peft_demo.txt",
}
# Merge dictionaries
configs_dict.update(model_configs)
configs = SimpleNamespace(**configs_dict)


# Data comes from https://huggingface.co/datasets/databricks/databricks-dolly-15k
def create_datasets(dataset_size=10, inference_file_path='inference_dataset.json', finetuning_file_path='finetuning_dataset.json'):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    data = []
    for row in dataset:
        if len(data) == dataset_size:
            break
        if ("open_qa" in row['category'] or "closed_qa" in row['category']) and len(row['context']) == 0:
            data.append(row['instruction'] + " " + row['response'])
    print("Number of datapoints:", len(data))
    with open(inference_file_path, 'w') as file:
        json.dump(data[:1], file)
    with open(finetuning_file_path, 'w') as file:
        json.dump(data, file, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    random.seed(configs.seed)

    create_datasets(inference_file_path=configs.prompt, finetuning_file_path=configs.finetuning_dataset)

    # Clear output file
    with open(configs.output_file, 'w') as file:
        file.write('')

    # Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
    ff.init(configs_dict)

    # Create the FlexFlow LLM
    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.base_model,
        data_type=ff_data_type,
        cache_path=configs.cache_path,
        refresh_cache=configs.refresh_cache,
        output_file=configs.output_file,
    )
    # Add inference and/or finetuning lora
    lora_inference_config = None
    lora_finetuning_config = None
    if len(configs.prompt) > 0:
        lora_inference_config = ff.LoraLinearConfig(
            llm.cache_path, configs.inference_peft_model_id
        )
        llm.add_peft(lora_inference_config)
    if len(configs.finetuning_dataset) > 0:
        lora_finetuning_config = ff.LoraLinearConfig(
            llm.cache_path,
            configs.inference_peft_model_id,
            trainable=True,
            optimizer_type=ff.OptimizerType.OPTIMIZER_TYPE_SGD,
            optimizer_kwargs={
                "learning_rate": 1.0,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
            },
        )
        llm.add_peft(lora_finetuning_config)

    # Compile the LLM for inference and load the weights into memory
    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        enable_peft_finetuning=(len(configs.finetuning_dataset) > 0),
        max_requests_per_batch=1,
        max_seq_length=256,
        max_tokens_per_batch=128,
    )

    llm.start_server()

    if len(configs.prompt) > 0:
        prompts = [s for s in json.load(open(configs.prompt))]
        inference_requests = [
            ff.Request(
                ff.RequestType.REQ_INFERENCE,
                prompt=prompt,
                max_sequence_length=128,
                peft_model_id=llm.get_ff_peft_id(lora_inference_config),
            )
            for prompt in prompts
        ]
        # llm.generate(inference_requests)

    # Finetuning
    if len(configs.finetuning_dataset) > 0:
        finetuning_request = ff.Request(
            ff.RequestType.REQ_FINETUNING,
            max_sequence_length=128,
            peft_model_id=llm.get_ff_peft_id(lora_finetuning_config),
            dataset_filepath=os.path.join(os.getcwd(), configs.finetuning_dataset),
            max_training_steps=100,
        )
        llm.generate(finetuning_request)

    llm.stop_server()