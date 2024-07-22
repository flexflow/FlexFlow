import json, random, subprocess
from datasets import load_dataset
from inference.python.peft_demo.demo import FlexFlowDemo
from types import SimpleNamespace
from huggingface_hub import HfFolder
import os
import flexflow.serve as ff
import matplotlib.pyplot as plt


def create_datasets(finetune_dataset_size=2, inference_file_path='inference_dataset.json', finetuning_file_path='finetuning_dataset.json'):
    """Creates the inference and finetuning datasets according to the data from https://huggingface.co/datasets/databricks/databricks-dolly-15k.
    Only the 'open_qa' and 'closed_qa' prompts without context are kept.
    The datasets are saved into the files given as arguments.

    Keyword arguments:
    dataset_size -- the number of prompts to consider
    inference_file_path -- the file in which to save the inference data
    finetuning_file_path -- the file in which to save the finetuning data
    """
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    inference_data = []
    finetuning_data = []
    for row in dataset:
        if len(finetuning_data) == finetune_dataset_size:
            break
        if ("open_qa" in row['category'] or "closed_qa" in row['category']) and len(row['context']) == 0:
            inference_data.append(row['instruction'])
            finetuning_data.append(row['instruction'] + " " + row['response'])
    with open(inference_file_path, 'w') as file:
        json.dump(inference_data[:1], file)
    with open(finetuning_file_path, 'w') as file:
        json.dump(finetuning_data[:1], file, indent=2, separators=(',', ': '))


configs_dict = {
    "num_gpus": 4,
    "memory_per_gpu": 14000,
    "zero_copy_memory_per_node": 40000,
    "num_cpus": 4,
    "legion_utility_processors": 4,
    "data_parallelism_degree": 1,
    "tensor_parallelism_degree": 1,
    "pipeline_parallelism_degree": 4,
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
    "max_requests_per_batch": 1,
    "max_sequence_length": 256,
    "max_tokens_per_batch": 128,
    "max_training_steps": 10,
    "seed": 42,
}
model_configs = {
    "base_model": "meta-llama/Meta-Llama-3-8B",
    "inference_peft_model_id": "goliaro/llama-3-8b-lora",
    "finetuning_peft_model_id": "flechman/llama-3-8b-lora-dolly",
    "cache_path": os.environ.get("FF_CACHE_PATH", ""),
    "refresh_cache": False,
    "full_precision": True,
    # relative paths
    "inference_dataset": "inference_dataset.json",
    "finetuning_dataset": "finetuning_dataset.json",
    "output_file": "peft_demo.txt",
}
generation_configs = {
    "do_sample": False,
    "temperature": 0.9,
    "topp": 0.8,
    "topk": 1,
}
finetuning_configs = {
    "learning_rate": 1.0,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
}
# Merge dictionaries
configs_dict.update(model_configs)
configs_dict.update(generation_configs)
configs_dict.update(finetuning_configs)


random.seed(configs_dict["seed"])

create_datasets(inference_file_path=configs_dict["inference_dataset"], 
                finetuning_file_path=configs_dict["finetuning_dataset"])

configs = SimpleNamespace(**configs_dict)

# Clear output file
with open(configs.output_file, 'w') as file:
    file.write('')

# Download base and peft inference models
args = [configs.inference_peft_model_id, '--base_model_name', configs.base_model]
hf_token = input("Please enter your HuggingFace personal access token: ")
subprocess.run(['huggingface-cli', 'login', '--token', hf_token])
subprocess.run(['python', '../../utils/download_peft_model.py'] + args)


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
if len(configs.inference_dataset) > 0:
    lora_inference_config = ff.LoraLinearConfig(
        llm.cache_path, 
        configs.inference_peft_model_id,
        base_model_name_or_path=configs.base_model
    )
    llm.add_peft(lora_inference_config)
if len(configs.finetuning_dataset) > 0:
    lora_finetuning_config = ff.LoraLinearConfig(
        llm.cache_path,
        configs.finetuning_peft_model_id,
        trainable=True,
        init_lora_weights=True,
        rank=16,
        lora_alpha=16.0,
        target_modules = ["down_proj"],
        base_model_name_or_path=configs.base_model,
        optimizer_type=ff.OptimizerType.OPTIMIZER_TYPE_SGD,
        optimizer_kwargs={
            "learning_rate": configs.learning_rate,
            "momentum": configs.momentum,
            "weight_decay": configs.weight_decay,
            "nesterov": configs.nesterov,
        },
    )
    llm.add_peft(lora_finetuning_config)

# Compile the LLM for inference and load the weights into memory
generation_config = ff.GenerationConfig(
    do_sample=configs.do_sample,
    temperature=configs.temperature,
    topp=configs.topp,
    topk=configs.topk
)
enable_peft_finetuning = len(configs.finetuning_dataset) > 0
llm.compile(
    generation_config,
    enable_peft_finetuning=enable_peft_finetuning,
    max_requests_per_batch=configs.max_requests_per_batch+int(enable_peft_finetuning),
    max_seq_length=configs.max_sequence_length,
    max_tokens_per_batch=configs.max_tokens_per_batch,
)


llm.start_server()


prompts = [s for s in json.load(open(configs.inference_dataset))]
inference_requests = [
    ff.Request(
        ff.RequestType.REQ_INFERENCE,
        prompt=prompt,
        max_sequence_length=configs.max_sequence_length,
        peft_model_id=llm.get_ff_peft_id(lora_inference_config),
    )
    for prompt in prompts
]
inf_req_res_1 = llm.generate(inference_requests)


finetuning_request = ff.Request(
    ff.RequestType.REQ_FINETUNING,
    max_sequence_length=configs.max_sequence_length,
    peft_model_id=llm.get_ff_peft_id(lora_finetuning_config),
    dataset_filepath=os.path.join(os.getcwd(), configs.finetuning_dataset),
    max_training_steps=configs.max_training_steps,
)
ft_res = llm.generate([finetuning_request])


hf_token = input("Please enter your HuggingFace personal access token: ")
subprocess.run(['huggingface-cli', 'login', '--token', hf_token])
subprocess.run(['python', '../../utils/upload_peft_model.py'] + [configs.finetuning_peft_model_id])


lora_inference_config = ff.LoraLinearConfig(
    llm.cache_path, 
    configs.finetuning_peft_model_id,
    base_model_name_or_path=configs.base_model
)
llm.add_peft(lora_inference_config)

args = [configs.finetuning_peft_model_id, '--base_model_name', configs.base_model]
#hf_token = input("Please enter your HuggingFace personal access token: ")
subprocess.run(['huggingface-cli', 'login', '--token', hf_token])
subprocess.run(['python', '../../utils/download_peft_model.py'] + args)


prompts = [s for s in json.load(open(configs.inference_dataset))]
inference_requests = [
    ff.Request(
        ff.RequestType.REQ_INFERENCE,
        prompt=prompt,
        max_sequence_length=configs.max_sequence_length,
        peft_model_id=llm.get_ff_peft_id(lora_inference_config),
    )
    for prompt in prompts
]
inf_req_res_2 = llm.generate(inference_requests)


llm.stop_server()


print("==Inference result before finetuning: ", inf_req_res_1[0].output_text)
print("==Inference result after finetuning: ", inf_req_res_2[0].output_text)


epochs = list(range(configs_dict["max_training_steps"]))
loss_values = ft_res[0].finetuning_losses

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b')