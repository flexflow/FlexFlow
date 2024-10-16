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
            "zero_copy_memory_per_node": 10000,
            # optional parameters
            "num_cpus": 4,
            "legion_utility_processors": 4,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 2,
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
            "fusion": False,
        }
        model_configs = {
            # required parameters
            "base_model": "JackFram/llama-160m",
            "inference_peft_model_id": "goliaro/llama-160m-lora",
            "finetuning_peft_model_id": "goliaro/llama-160m-lora",
            # "base_model": "meta-llama/Meta-Llama-3-8B",
            # "inference_peft_model_id": "goliaro/llama-3-8b-lora",
            # "finetuning_peft_model_id": "goliaro/llama-3-8b-lora-dolly",
            # optional parameters
            "cache_path": os.environ.get("FF_CACHE_PATH", ""),
            "refresh_cache": False,
            "full_precision": True,
            "prompt": "",
            "finetuning_dataset": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../prompt/peft_dataset.json",
            ),
            "output_file": "",
        }
        # Merge dictionaries
        ff_init_configs.update(model_configs)
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
            llm.cache_path,
            configs.inference_peft_model_id,
            base_model_name_or_path=configs.base_model,
        )
        llm.add_peft(lora_inference_config)
    if len(configs.finetuning_dataset) > 0:
        # lora_finetuning_config = ff.LoraLinearConfig(
        #     llm.cache_path,
        #     configs.finetuning_peft_model_id,
        #     target_modules=["down_proj"],
        #     rank=16,
        #     lora_alpha=16,
        #     trainable=True,
        #     init_lora_weights=True,
        #     optimizer_type=ff.OptimizerType.OPTIMIZER_TYPE_SGD,
        # )
        lora_finetuning_config = ff.LoraLinearConfig(
            llm.cache_path,
            configs.inference_peft_model_id,
            trainable=True,
            base_model_name_or_path=configs.base_model,
            optimizer_type=ff.OptimizerType.OPTIMIZER_TYPE_SGD,
            optimizer_kwargs={
                "learning_rate": 0.001,
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
    enable_peft_finetuning = len(configs.finetuning_dataset) > 0
    llm.compile(
        generation_config,
        enable_peft_finetuning=enable_peft_finetuning,
        max_requests_per_batch=1 if not enable_peft_finetuning else 2,
        max_seq_length=256,
        max_tokens_per_batch=128,
    )

    llm.start_server()

    requests = []
    # Serving
    if len(configs.prompt) > 0:
        prompts = [s for s in json.load(open(configs.prompt))]
        inference_requests = [
            ff.Request(
                ff.RequestType.REQ_INFERENCE,
                prompt=prompt,
                max_new_tokens=128,
                peft_model_id=llm.get_ff_peft_id(lora_inference_config),
            )
            for prompt in prompts
        ]
        requests += inference_requests
    # Finetuning
    if len(configs.finetuning_dataset) > 0:
        finetuning_request = ff.Request(
            ff.RequestType.REQ_FINETUNING,
            peft_model_id=llm.get_ff_peft_id(lora_finetuning_config),
            dataset_filepath=configs.finetuning_dataset,
            max_training_steps=2,
        )
        requests.append(finetuning_request)

    results = llm.generate(requests)

    llm.stop_server()


if __name__ == "__main__":
    print("flexflow PEFT example")
    main()
