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
This script implements the usecase of prompt template upon FlexFlow.

Functionality:
1. FlexFlowLLM Class:
   - Initializes and configures FlexFlow.
   - Loads configurations from a file or uses default settings.
   - Compiles and starts the language model server for text generation.
   - Stops the server when operations are complete.

2. FF_LLM_wrapper Class:
   - Serves as a wrapper for FlexFlow.
   - Implements the necessary interface to interact with the LangChain library.

3. Main:
   - Initializes FlexFlow.
   - Compiles and starts the server with specific generation configurations.
   - Sets up a prompt template for generating responses to questions.
   - Use LLMChain to run the model and generate response.
   - Stops the FlexFlow server after generating the response.
"""

import flexflow.serve as ff
import argparse, json, os
from types import SimpleNamespace
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


class FlexFlowLLM:
    def __init__(self, config_file=""):
        self.configs = self.get_configs(config_file)
        ff.init(self.configs)
        self.llm = self.create_llm()
        self.ssms = self.create_ssms()

    def get_configs(self, config_file):
        # Load configurations from a file or use default settings
        if config_file and os.path.isfile(config_file):
            with open(config_file) as f:
                return json.load(f)
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
            
    def create_llm(self):
        configs = SimpleNamespace(**self.configs)
        ff_data_type = ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
        llm = ff.LLM(
            configs.llm_model,
            data_type=ff_data_type,
            cache_path=configs.cache_path,
            refresh_cache=configs.refresh_cache,
            output_file=configs.output_file,
        )
        return llm
    
    def create_ssms(self):
        # Create the SSMs
        configs = SimpleNamespace(**self.configs)
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
        return ssms
    
    def compile_and_start(self, generation_config, max_requests_per_batch, max_seq_length, max_tokens_per_batch):
        
        # Compile the SSMs for inference and load the weights into memory
        for ssm in self.ssms:
            ssm.compile(
                generation_config,
                max_requests_per_batch,
                max_seq_length,
                max_tokens_per_batch,
            )
            
        # Compile the LLM for inference and load the weights into memory
        self.llm.compile(
            generation_config, 
            max_requests_per_batch, 
            max_seq_length, 
            max_tokens_per_batch,
            ssms = self.ssms
        )
        self.llm.start_server()

    def generate(self, prompt):
        results = self.llm.generate(prompt)
        if isinstance(results, list):
            result_txt = results[0].output_text.decode('utf-8')
        else:
            result_txt = results.output_text.decode('utf-8')
        return result_txt
    
    def stop_server(self):
        self.llm.stop_server()

    def __enter__(self):
        return self.llm.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.llm.__exit__(exc_type, exc_value, traceback)

class FF_LLM_wrapper(LLM):
    flexflow_llm: FlexFlowLLM

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = self.flexflow_llm.generate(prompt)
        return response


if __name__ == "__main__":
    # initialization
    ff_llm = FlexFlowLLM()

    # compile and start server
    gen_config = ff.GenerationConfig(do_sample=False, temperature=0.9, topp=0.8, topk=1)
    ff_llm.compile_and_start(
        gen_config, 
        max_requests_per_batch=1, 
        max_seq_length=256, 
        max_tokens_per_batch=64
    )

    # the wrapper class serves as the 'Model' in LCEL 
    ff_llm_wrapper = FF_LLM_wrapper(flexflow_llm=ff_llm)
    
    # USE CASE 1: Prompt Template
    template = """Question: {question}
                    Answer: Let's think step by step."""
    
    # Build prompt template and langchain
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=ff_llm_wrapper)

    question = "Who was the US president in the year the first Pokemon game was released?"
    print(llm_chain.run(question))
    
    # stop the server
    ff_llm.stop_server()


