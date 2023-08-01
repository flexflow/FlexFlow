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
import os, json
from types import SimpleNamespace


def parse_configs():
    os.chdir(os.path.dirname(__file__))
    configs = {
        "llm_model": "decapoda-research/llama-7b-hf",
        "llm_weight": "",
        "llm_tokenizer": "",
        "clean_model_cache": False,
        "full_precision": False,
        "ssms": [
            {
                "ssm_model": "JackFram/llama-160m",
                "ssm_weight": "",
                "ssm_tokenizer": "",
                "clean_model_cache": False,
                "full_precision": False,
            },
            {
                "ssm_model": "facebook/opt-125m",
                "ssm_weight": "",
                "ssm_tokenizer": "",
                "clean_model_cache": False,
                "full_precision": False,
            },
        ],
        "prompt": "../prompt/test.json",
        "output_file": "",
    }
    return SimpleNamespace(**configs)


def top_level_task():
    configs = parse_configs()

    # Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
    ff.init(
        {
            # required arguments
            "num_gpus": 4,
            "memory_per_gpu": 14000,
            "zero_copy_memory_per_gpu": 30000,
            # optional arguments
            "num_cpus": 4,
            "legion_utility_processors": 4,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 4,
            "offload": False,
            "offload_reserve_space_size": 1024**2,
            "use_4bit_quantization": False,
            "use_8bit_quantization": False,
            "profiling": False,
            "fusion": True,
        }
    )

    # Create the FlexFlow LLM
    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.llm_model,
        data_type=ff_data_type,
        tokenizer_path=configs.llm_tokenizer,
        weights_path=configs.llm_weight,
        clean_cache=configs.clean_model_cache,
        output_file=configs.output_file,
    )

    # Create the SSMs
    ssms = []
    for ssm_config in configs.ssms:
        ssm_config = SimpleNamespace(**ssm_config)
        ff_data_type = (
            ff.DataType.DT_FLOAT if ssm_config.full_precision else ff.DataType.DT_HALF
        )
        ssm = ff.LLM(
            ssm_config.ssm_model,
            data_type=ff_data_type,
            tokenizer_path=ssm_config.ssm_tokenizer,
            weights_path=ssm_config.ssm_weight,
            clean_cache=ssm_config.clean_model_cache,
            output_file=configs.output_file,
        )
        ssms.append(ssm)

    # Create the sampling configs
    sampling_config = ff.SamplingConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )

    # Compile the SSMs for inference and load the weights into memory
    for ssm in ssms:
        ssm.compile(
            ff.InferenceMode.BEAM_SEARCH_MODE,
            sampling_config,
            max_batch_size=1,
            max_seq_length=256,
            max_tokens_per_batch=64,
        )

    # Compile the LLM for inference and load the weights into memory
    llm.compile(
        ff.InferenceMode.TREE_VERIFY_MODE,
        sampling_config,
        max_batch_size=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
        ssms=ssms,
    )

    # Generation begins!
    if len(configs.prompt) > 0:
        prompts = [s for s in json.load(open(configs.prompt))]
        results = llm.generate(prompts)
    else:
        result = llm.generate("Here are some travel tips for Tokyo:\n")


if __name__ == "__main__":
    print("flexflow inference (speculative inference)")
    top_level_task()
