#!/usr/bin/env python
import os, json

# Base configs dictionaries
ff_init_configs = {
    # required parameters
    "num_gpus": 4,
    "memory_per_gpu": 14000,
    "zero_copy_memory_per_node": 40000,
    # optional parameters
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
    "benchmarking": False,
    "inference_debugging": False,
    "fusion": True,
}
llm_configs = {
    # required parameters
    "llm_model": "tiiuae/falcon-7b",
    # optional parameters
    "cache_path": os.environ.get("FF_CACHE_PATH", ""),
    "refresh_cache": False,
    "full_precision": True,
    "prompt": "",
    "output_file": "",
}
ssm_configs = {
    "ssms": [
        {
            # required ssm parameter
            "ssm_model": "JackFram/llama-160m",
            # optional ssm parameters
            "cache_path": "",
            "refresh_cache": False,
            "full_precision": False,
        },
    ]
}
# Merge dictionaries
ff_init_configs.update(llm_configs)

# Test parameters to fill in
llama_models = ["meta-llama/Llama-2-7b-hf", "JackFram/llama-160m"]
opt_models = ["facebook/opt-6.7b", "facebook/opt-125m"]
falcon_models = [
    "tiiuae/falcon-7b",
]
mpt_models = [
    "mosaicml/mpt-7b",
]
# starcoder_models = ["bigcode/starcoderbase-7b",]
parallelism_settings = [(1, 4), (2, 2), (4, 1)]

# The paths below should be with respect to the folder from which the tests are launched (FF_HOME/tests/inference)
prompt_file = "../../inference/prompt/test.json"
output_folder = "../../inference/output"

# Change working dir to folder storing this script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# Generate incremental decoding configs
all_models = llama_models + opt_models + falcon_models + mpt_models
for model_name in all_models:
    for full_precision in (True, False):
        for parallelism_degrees in parallelism_settings:
            tp, pp = parallelism_degrees

            # Tensor parallelism not supported by small Falcon model atm
            if tp > 1 and ("falcon" in model_name):
                continue
            # skip tp=4 for big models
            if tp > 2 and ("7b" in model_name or "6.7b" in model_name):
                continue

            # Run Falcon only in full precision, Starcoder only in half precision
            if (not full_precision and "falcon" in model_name) or (full_precision and "starcoder" in model_name):
                continue

            _, after_slash = model_name.rsplit("/", maxsplit=1)
            filename = (
                "incr_dec-"
                + "python-"
                + after_slash.lower()
                + ("-full_prec-" if full_precision else "-half_prec-")
                + f"{tp}_tp_{pp}_pp"
            )
            test_configs_file = "./" + filename + ".json"
            output_file = os.path.join(output_folder, filename + ".txt")

            ff_init_configs["tensor_parallelism_degree"] = tp
            ff_init_configs["pipeline_parallelism_degree"] = pp
            ff_init_configs["llm_model"] = model_name
            ff_init_configs["full_precision"] = full_precision
            ff_init_configs["output_file"] = output_file
            ff_init_configs["prompt"] = prompt_file

            with open(test_configs_file, "w+") as outfile:
                json.dump(ff_init_configs, outfile, indent=4)

# Generate speculative inference configs
model_pairs = [llama_models, opt_models]
for model_pair in model_pairs:
    for full_precision in (True, False):
        for parallelism_degrees in parallelism_settings:
            big_model, small_model = model_pair
            tp, pp = parallelism_degrees

            # Skip fully tp tests
            if tp > 2:
                continue

            _, after_slash = big_model.rsplit("/", maxsplit=1)
            filename = (
                "spec_infer-"
                + "python-"
                + after_slash.lower()
                + ("-full_prec-" if full_precision else "-half_prec-")
                + f"{tp}_tp_{pp}_pp"
            )
            test_configs_file = "./" + filename + ".json"
            output_file = os.path.join(output_folder, filename + ".txt")

            ff_init_configs["tensor_parallelism_degree"] = tp
            ff_init_configs["pipeline_parallelism_degree"] = pp
            ff_init_configs["llm_model"] = big_model
            ff_init_configs["full_precision"] = full_precision
            ff_init_configs["output_file"] = output_file
            ff_init_configs["prompt"] = prompt_file

            ssm_configs["ssms"][0]["ssm_model"] = small_model
            ssm_configs["ssms"][0]["full_precision"] = full_precision
            ff_init_configs.update(ssm_configs)

            with open(test_configs_file, "w+") as outfile:
                json.dump(ff_init_configs, outfile, indent=4)
