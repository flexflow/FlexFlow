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
            "benchmarking": False,
            "inference_debugging": False,
            "fusion": True,
        }
        llm_configs = {
            # required llm arguments
            "llm_model": "meta-llama/Llama-2-7b-hf",
            # optional llm parameters
            "cache_path": os.environ.get("FF_CACHE_PATH", ""),
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


def main():
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
            max_tokens_per_batch=64,
        )

    # Compile the LLM for inference and load the weights into memory
    llm.compile(
        generation_config,
        max_requests_per_batch=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
        ssms=ssms,
    )
    
    llm.start_server()

    if len(configs.prompt) > 0:
        prompts = [s for s in json.load(open(configs.prompt))]
        results = llm.generate(prompts)
    else:
        result = llm.generate("Three tips for staying healthy are: ")
        
    llm.stop_server()

if __name__ == "__main__":
    print("flexflow inference example (speculative inference)")
    main()
