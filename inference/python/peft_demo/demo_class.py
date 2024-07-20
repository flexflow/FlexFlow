import flexflow.serve as ff
import json, os, warnings
from types import SimpleNamespace
import random


class FlexFlowDemo(object):

    def __init__(self, configs_dict):
        self.configs_dict = configs_dict
        self.configs = SimpleNamespace(**configs_dict)
        self.llm = None
        self.server_started = False
        self.server_stopped = False

        # Clear output file
        with open(self.configs.output_file, 'w') as file:
            file.write('')

    def initialize_flexflow(self):
        if self.llm is None:
            # Initialize the FlexFlow runtime. ff.init() takes a dictionary or the path to a JSON file with the configs
            ff.init(self.configs_dict)

            # Create the FlexFlow LLM
            ff_data_type = (
                ff.DataType.DT_FLOAT if self.configs.full_precision else ff.DataType.DT_HALF
            )
            self.llm = ff.LLM(
                self.configs.base_model,
                data_type=ff_data_type,
                cache_path=self.configs.cache_path,
                refresh_cache=self.configs.refresh_cache,
                output_file=self.configs.output_file,
            )
            # Add inference and/or finetuning lora
            self.lora_inference_config = None
            self.lora_finetuning_config = None
            if len(self.configs.inference_dataset) > 0:
                self.lora_inference_config = ff.LoraLinearConfig(
                    self.llm.cache_path, 
                    self.configs.inference_peft_model_id,
                    base_model_name_or_path=self.configs.base_model
                )
                self.llm.add_peft(self.lora_inference_config)
            if len(self.configs.finetuning_dataset) > 0:
                self.lora_finetuning_config = ff.LoraLinearConfig(
                    self.llm.cache_path,
                    self.configs.inference_peft_model_id,
                    trainable=True,
                    base_model_name_or_path=self.configs.base_model,
                    optimizer_type=ff.OptimizerType.OPTIMIZER_TYPE_SGD,
                    optimizer_kwargs={
                        "learning_rate": self.configs.learning_rate,
                        "momentum": self.configs.momentum,
                        "weight_decay": self.configs.weight_decay,
                        "nesterov": self.configs.nesterov,
                    },
                )
                self.llm.add_peft(self.lora_finetuning_config)

            # Compile the LLM for inference and load the weights into memory
            generation_config = ff.GenerationConfig(
                do_sample=self.configs.do_sample,
                temperature=self.configs.temperature,
                topp=self.configs.topp,
                topk=self.configs.topk
            )
            enable_peft_finetuning = len(self.configs.finetuning_dataset) > 0
            self.llm.compile(
                generation_config,
                enable_peft_finetuning=enable_peft_finetuning,
                max_requests_per_batch=self.configs.max_requests_per_batch+int(enable_peft_finetuning),
                max_seq_length=self.configs.max_sequence_length,
                max_tokens_per_batch=self.configs.max_tokens_per_batch,
            )
        else:
            warnings.warn("FlexFlow has already been initialized. The behavior of the program from now on is undefined.")

    def start_server(self):
        if self.llm is None:
            raise Exception("FlexFlow has not been initialized.")
        if not self.server_started and not self.server_stopped:
            self.llm.start_server()
            self.server_started = True

    def stop_server(self):
        if self.llm is None:
            raise Exception("FlexFlow has not been initialized.")
        if self.server_started and not self.server_stopped:
            self.llm.stop_server()
            self.server_stopped = True

    def generate_inference(self):
        if self.llm is None:
            raise Exception("FlexFlow has not been initialized.")
        if not self.server_started:
            raise Exception("Server has not started.")
        if self.server_stopped:
            raise Exception("Server stopped.")
        
        if len(self.configs.inference_dataset) > 0:
            prompts = [s for s in json.load(open(self.configs.inference_dataset))]
            inference_requests = [
                ff.Request(
                    ff.RequestType.REQ_INFERENCE,
                    prompt=prompt,
                    max_sequence_length=self.configs.max_sequence_length,
                    peft_model_id=self.llm.get_ff_peft_id(self.lora_inference_config),
                )
                for prompt in prompts
            ]
            self.llm.generate(inference_requests)

    def generate_finetuning(self):
        if self.llm is None:
            raise Exception("FlexFlow has not been initialized.")
        if not self.server_started:
            raise Exception("Server has not started.")
        if self.server_stopped:
            raise Exception("Server stopped.")

        reqs = []
        if len(self.configs.finetuning_dataset) > 0:
            finetuning_request = ff.Request(
                ff.RequestType.REQ_FINETUNING,
                max_sequence_length=self.configs.max_sequence_length,
                peft_model_id=self.llm.get_ff_peft_id(self.lora_finetuning_config),
                dataset_filepath=os.path.join(os.getcwd(), self.configs.finetuning_dataset),
                max_training_steps=self.configs.max_training_steps,
            )
            reqs += [finetuning_request]
        if len(self.configs.inference_dataset) > 0:
            prompts = [s for s in json.load(open(self.configs.inference_dataset))]
            inference_requests = [
                ff.Request(
                    ff.RequestType.REQ_INFERENCE,
                    prompt=prompt,
                    max_sequence_length=self.configs.max_sequence_length,
                    peft_model_id=self.llm.get_ff_peft_id(self.lora_inference_config),
                )
                for prompt in prompts
            ]
            reqs += inference_requests
        self.llm.generate(reqs)

def main():
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
        "max_training_steps": 4,
        "seed": 42,
    }
    model_configs = {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "inference_peft_model_id": "goliaro/llama-3-8b-lora",
        "finetuning_peft_model_id": "goliaro/llama-3-8b-lora",
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

    demo = FlexFlowDemo(configs_dict)

    demo.initialize_flexflow()
    demo.start_server()
    demo.generate_finetuning()
    demo.generate_inference()
    demo.stop_server()

if __name__ == "__main__":
    main()