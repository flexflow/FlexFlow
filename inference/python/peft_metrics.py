# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flexflow.serve as ff
import argparse, json, os
from types import SimpleNamespace
import time
import subprocess
import psutil
import time
import json


def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip()
        lines = output.split('\n')
        
        total_gpu_utilization = 0.0
        total_memory_used = 0.0
        num_gpus = len(lines)
        
        for line in lines:
            try:
                gpu_utilization, memory_used = line.split(', ')
                total_gpu_utilization += float(gpu_utilization)
                total_memory_used += float(memory_used)
            except ValueError:
                print("Error parsing line:", line)
                num_gpus -= 1  # Adjust num_gpus in case of parsing failure
        
        # Handle division by zero if no GPUs are found or parsed successfully
        if num_gpus > 0:
            avg_gpu_utilization = total_gpu_utilization / num_gpus
            avg_memory_used = total_memory_used / num_gpus
        else:
            avg_gpu_utilization = 0.0
            avg_memory_used = 0.0
        
        
        # print(f"GPU Utilization: {avg_gpu_utilization}%")
        # print(f"Memory Used: {avg_memory_used} MiB")
        
        return avg_gpu_utilization, avg_memory_used
    except Exception as e:
        print(f"Failed to get GPU utilization: {e}")
        return 0, 0



def get_cpu_utilization():
    # Gets the system-wide CPU utilization
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    # Gets the system-wide memory usage
    memory_info = psutil.virtual_memory()
    return memory_info.used / (1024 * 1024)  # Convert to MB

def monitor_resources(start_time, interval=5, duration=60):
    """
    Monitors and collects resource usage metrics over a specified duration and interval.
    
    :param start_time: The time when the monitoring started, to calculate total duration.
    :param interval: Time in seconds between each metric collection.
    :param duration: Total duration to monitor resources.
    :return: A dictionary containing the collected metrics.
    """
    metrics = {
        'max_gpu_utilization': 0,
        'max_memory_usage_gpu': 0,
        'cpu_utilization': [],
        'peak_memory_usage_system': 0,
    }
    
    while True:
        current_time = time.time()
        if current_time - start_time > duration:
            break
        
        gpu_utilization, memory_usage_gpu = get_gpu_utilization()
        cpu_utilization = get_cpu_utilization()
        memory_usage_system = get_memory_usage()
        
        metrics['max_gpu_utilization'] = max(metrics['max_gpu_utilization'], gpu_utilization)
        metrics['max_memory_usage_gpu'] = max(metrics['max_memory_usage_gpu'], memory_usage_gpu)
        metrics['cpu_utilization'].append(cpu_utilization)
        metrics['peak_memory_usage_system'] = max(metrics['peak_memory_usage_system'], memory_usage_system)
        
        time.sleep(interval)
    
    return metrics

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config-file",
        help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--publish-peft-with-id", 
        help="The Hugging Face model ID to upload the trained model with",
        type=str, 
        default=""
    )

    args = parser.parse_args()
    publish_peft_with_id = args.publish_peft_with_id
    if len(publish_peft_with_id) == 0:
        print(
            "Please pass a --publish-peft-with-id if you want to upload the trained model"
        )
    else:
        print(f"The trained model will be uploaded with id: {publish_peft_with_id}")
        
    # Load configs from JSON file (if specified)
    if len(args.config_file) > 0:
        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f"Config file {args.config_file} not found.")
        try:
            with open(args.config_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print("JSON format error:")
            print(e)
    else:
        # Define sample configs
        ff_init_configs = {
            # required parameters
            "num_gpus": 1,
            "memory_per_gpu": 8192,
            "zero_copy_memory_per_node": 12000,
            # optional parameters
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
            "inference_debugging": True,
            "fusion": True,
        }
        model_configs = {
            # required parameters
            "base_model": "JackFram/llama-160m",
            "peft_model_ids": [
                "goliaro/llama-160m-lora-full",
            ],
            # optional parameters
            "cache_path": "~/.cache/flexflow",
            "refresh_cache": False,
            "full_precision": False,
            "prompt": "",
            "finetuning_dataset": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../prompt/peft.json"
                # peft.json is a sample dataset for finetuning, should contain a list of strings
            ),
            "output_file": ""
        }
        # Merge dictionaries
        ff_init_configs.update(model_configs)
        ff_init_configs["publish_peft_with_id"] = publish_peft_with_id
        return ff_init_configs


def main():
    start_time = time.time()
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)

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
        output_file=configs.output_file
    )
    for peft_model_id in configs.peft_model_ids:
        llm.add_peft(peft_model_id)

    # Compile the LLM for inference and load the weights into memory
    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        max_requests_per_batch=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
    )
    
    resource_metrics = monitor_resources(start_time, interval=5, duration=360)

    llm.start_server()
    
    print(f"LLM model class is: {llm.model_class}")

    requests = []
    # Serving
    if len(configs.prompt) > 0:
        prompts = [s for s in json.load(open(configs.prompt))]
        inference_requests = [
            ff.Request(
                ff.RequestType.REQ_INFERENCE, prompt=prompt, max_sequence_length=128
            )
            for prompt in prompts
        ]
        requests += inference_requests
    # Finetuning
    if len(configs.finetuning_dataset) > 0:
        for peft_model_id in configs.peft_model_ids:
            finetuning_request = ff.Request(
                ff.RequestType.REQ_FINETUNING,
                max_sequence_length=128,
                peft_model_id=llm.get_ff_peft_id(peft_model_id),
                dataset_filepath=configs.finetuning_dataset,
            )
            requests.append(finetuning_request)
            
    # use the (finetuned) llm to generate some responses
    llm.generate(requests)
    
    # After finishing the main workload, print the collected metrics.
    avg_cpu_utilization = sum(resource_metrics['cpu_utilization']) / len(resource_metrics['cpu_utilization'])
    print(f"Max GPU Utilization: {resource_metrics['max_gpu_utilization']}%")
    print(f"Max GPU Memory Usage: {resource_metrics['max_memory_usage_gpu']} MiB")
    print(f"Average CPU Utilization: {avg_cpu_utilization}%")
    print(f"Peak System Memory Usage: {resource_metrics['peak_memory_usage_system']} MiB")

    
    llm.stop_server()
    
    # upload the model back to huggingface after finetuning
    # the model format would be converted from flexflow format back to huggingface format
    if len(configs.publish_peft_with_id) > 0:
        print(
            f"Done training! Uploading the model to HF hub with id: {configs.publish_peft_with_id}..."
        )
        llm.upload_peft_model(configs.publish_peft_with_id, private=True)
    

if __name__ == "__main__":
    print("flexflow PEFT example")
    
    main()