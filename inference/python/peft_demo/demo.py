import flexflow.serve as ff
import argparse, json, os
from types import SimpleNamespace
from datasets import load_dataset
import random


def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config-file",
        help="The path to a JSON file with the configs.",
        type=str,
        default="",
        required=True,
    )
    args = parser.parse_args()

    # Load configs from JSON file
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file {args.config_file} not found.")
    try:
        with open(args.config_file) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print("JSON format error:")
        print(e)

def init_llm_co_serving(configs_dict, configs):
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

    # Add the different PEFT models to finetune
    for peft_model_id in configs.peft_model_ids:
        llm.add_peft(peft_model_id)

    # Compile the LLM for inference and load the weights into memory
    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        enable_peft_finetuning = (len(configs.finetuning_dataset) > 0),
        max_requests_per_batch=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
    )

# Data comes from https://huggingface.co/datasets/databricks/databricks-dolly-15k
def import_dataset():
    inference_percentage = 0.6
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    data = []
    for i,row in enumerate(dataset):
        if len(row['context']) == 0:
            data.append((row['instruction'],row['response']))
    inference_prompts = []
    finetuning_prompts = []
    for d in data:
        if random.random() <= inference_percentage:
            inference_prompts.append(d[0])
        else:
            finetuning_prompts.append(d)
    return inference_prompts, finetuning_prompts

    
if __name__ == "__main__":
    print("Co-Serving Demo")
    # Import config parameters
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)
    random.seed(configs.seed)
    # Import inference dataset
    # Import finetuning dataset
    inference_prompts, finetuning_prompts = import_dataset()
    # Initialize Llama2 lora model
    llm = init_llm_co_serving(configs_dict, configs)
    llm.start_server()
    requests = []
    # Prepare inference requests
    inference_requests = [
        ff.Request(
            ff.RequestType.REQ_INFERENCE, 
            prompt=prompt, 
            max_sequence_length=configs.max_sequence_length
        )
        for prompt in inference_prompts
    ]
    requests += inference_requests
    # Prepare finetuning requests
    for peft_model_id in configs.peft_model_ids:
        finetuning_request = ff.Request(
            ff.RequestType.REQ_FINETUNING,
            max_sequence_length=configs.max_sequence_length,
            peft_model_id=llm.get_ff_peft_id(peft_model_id),
            dataset=finetuning_prompts,
        )
        requests.append(finetuning_request)
    # Jointly serve inference and finetuning requests
    llm.generate(requests, max_length=configs.max_sequence_length)
    llm.stop_server()
    # Show statistics and metrics of the system
    ## Show difference in loss on test dataset with finetuned and non-finetuned to prove that it works
    ## Show compute resources utilized + other metrics
    ## Compare with compute resources utilized without co-serving