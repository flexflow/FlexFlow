import flexflow.serve as ff
import json, os, warnings
from types import SimpleNamespace


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
            if len(self.configs.prompt) > 0:
                self.lora_inference_config = ff.LoraLinearConfig(
                    self.llm.cache_path, self.configs.inference_peft_model_id
                )
                self.llm.add_peft(self.lora_inference_config)
            if len(self.configs.finetuning_dataset) > 0:
                self.lora_finetuning_config = ff.LoraLinearConfig(
                    self.llm.cache_path,
                    self.configs.inference_peft_model_id,
                    trainable=True,
                    optimizer_type=ff.OptimizerType.OPTIMIZER_TYPE_SGD,
                    optimizer_kwargs={
                        "learning_rate": 1.0,
                        "momentum": 0.0,
                        "weight_decay": 0.0,
                        "nesterov": False,
                    },
                )
                self.llm.add_peft(self.lora_finetuning_config)

            # Compile the LLM for inference and load the weights into memory
            generation_config = ff.GenerationConfig(
                do_sample=False, temperature=0.9, topp=0.8, topk=1
            )
            self.llm.compile(
                generation_config,
                enable_peft_finetuning=(len(self.configs.finetuning_dataset) > 0),
                max_requests_per_batch=1,
                max_seq_length=256,
                max_tokens_per_batch=128,
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
        
        if len(self.configs.prompt) > 0:
            prompts = [s for s in json.load(open(self.configs.prompt))]
            inference_requests = [
                ff.Request(
                    ff.RequestType.REQ_INFERENCE,
                    prompt=prompt,
                    max_sequence_length=128,
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

        if len(self.configs.finetuning_dataset) > 0:
            finetuning_request = ff.Request(
                ff.RequestType.REQ_FINETUNING,
                max_sequence_length=128,
                peft_model_id=self.llm.get_ff_peft_id(self.lora_finetuning_config),
                dataset_filepath=os.path.join(os.getcwd(), self.configs.finetuning_dataset),
                max_training_steps=100,
            )
            self.llm.generate([finetuning_request])