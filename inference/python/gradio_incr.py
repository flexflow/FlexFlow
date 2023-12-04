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
            # required parameters
            "llm_model": "tiiuae/falcon-7b",
            # optional parameters
            "cache_path": "",
            "refresh_cache": False,
            "full_precision": False,
            "prompt": "",
            "output_file": "",
        }
        # Merge dictionaries
        ff_init_configs.update(llm_configs)
        return ff_init_configs


def generate_response(user_input):
    result = llm.generate(user_input)
    return result.output_text.decode('utf-8')


def main():
    
    global llm
    
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)

    ff.init(configs_dict)

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

    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        max_requests_per_batch=1,
        max_seq_length=256,
        max_tokens_per_batch=64,
    )

    iface = gr.Interface(
        fn=generate_response, 
        inputs="text", 
        outputs="text"
    )
    llm.start_server()

    iface.launch()

    llm.stop_server()

if __name__ == "__main__":
    print("flexflow inference example with gradio interface")
    main()