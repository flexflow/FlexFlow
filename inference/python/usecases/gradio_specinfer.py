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

"""
Functionality:
1. Configuration Handling:
   - Parses command-line arguments to get a configuration file path.
   - Loads configuration settings from a JSON file if provided, or uses default settings.

2. FlexFlow Model Initialization:
   - Initializes FlexFlow with the provided or default configurations.
   - Sets up the LLM with the specified model and configurations.
   - Compiles the model with generation settings and starts the FlexFlow server.

3. Gradio Interface Setup:
   - Defines a function to generate responses based on user input using FlexFlow.
   - Sets up a Gradio Chat Interface to interact with the model in a conversational format.

4. Main Execution:
   - Calls the main function to initialize configurations, set up the FlexFlow LLM, and launch the Gradio interface.
   - Stops the FlexFlow server after the Gradio interface is closed.

Usage:
1. Run the script with an optional configuration file argument for custom settings.
2. Interact with the FlexFlow model through the Gradio web interface.
3. Enter text inputs to receive generated responses from the model.
4. The script will stop the FlexFlow server automatically upon closing the Gradio interface.
"""

"""
TODO: fix current issue: model init is stuck at "prepare next batch init" and "prepare next batch verify"
"""

import gradio as gr
import flexflow.serve as ff
import argparse, json, os
from types import SimpleNamespace

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config-file",
        help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
        type=str,
        default="",
    )
    args = parser.parse_args()

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
            "num_gpus": 2,
            "memory_per_gpu": 14000,
            "zero_copy_memory_per_node": 40000,
            # optional parameters
            "num_cpus": 4,
            "legion_utility_processors": 4,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 2,
            "offload": False,
            "offload_reserve_space_size": 1024**2,
            "use_4bit_quantization": False,
            "use_8bit_quantization": False,
            "profiling": False,
            "inference_debugging": False,
            "fusion": True,
        }
        llm_configs = {
            # required llm arguments
            "llm_model": "meta-llama/Llama-2-7b-hf",
            # optional llm parameters
            "cache_path": "",
            "refresh_cache": False,
            "full_precision": False,
            "ssms": [
                {
                    # required ssm parameter
                    "ssm_model": "JackFram/llama-160m",
                    # optional ssm parameters
                    "cache_path": "",
                    "refresh_cache": False,
                    "full_precision": False,
                }
            ],
            # "prompt": "",
            "output_file": "",
        }
        # Merge dictionaries
        ff_init_configs.update(llm_configs)
        return ff_init_configs


# def generate_response(user_input):
#     result = llm.generate(user_input)
#     return result.output_text.decode('utf-8')

def generate_response(message, history):
    user_input = message 
    results = llm.generate(user_input)
    if isinstance(results, list):
        result_txt = results[0].output_text.decode('utf-8')
    else:
        result_txt = results.output_text.decode('utf-8')
    return result_txt

def main():
    
    global llm
    
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)

    # Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
    ff.init(configs_dict)

    # Create the FlexFlow LLM
    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.llm_model,
        data_type=ff_data_type,
        cache_path=configs.cache_path,
        refresh_cache=configs.refresh_cache,
        output_file=configs.output_file,
    )

    # Create the SSMs
    ssms = []
    for ssm_config in configs.ssms:
        ssm_config = SimpleNamespace(**ssm_config)
        ff_data_type = (
            ff.DataType.DT_FLOAT if ssm_config.full_precision else ff.DataType.DT_HALF
        )
        ssm = ff.SSM(
            ssm_config.ssm_model,
            data_type=ff_data_type,
            cache_path=ssm_config.cache_path,
            refresh_cache=ssm_config.refresh_cache,
            output_file=configs.output_file,
        )
        ssms.append(ssm)

    # Create the sampling configs
    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )

    # Compile the SSMs for inference and load the weights into memory
    for ssm in ssms:
        ssm.compile(
            generation_config,
            max_requests_per_batch=1,
            max_seq_length=256,
            max_tokens_per_batch=256,
        )

    # Compile the LLM for inference and load the weights into memory
    llm.compile(
        generation_config,
        max_requests_per_batch=1,
        max_seq_length=256,
        max_tokens_per_batch=256,
        ssms=ssms,
    )
    
    # # interface version 1
    # iface = gr.Interface(
    #     fn=generate_response, 
    #     inputs="text", 
    #     outputs="text"
    # )
    
    # interface version 2
    iface = gr.ChatInterface(fn=generate_response)
    llm.start_server()
    iface.launch()
    llm.stop_server()

if __name__ == "__main__":
    print("flexflow inference example with gradio interface")
    main()