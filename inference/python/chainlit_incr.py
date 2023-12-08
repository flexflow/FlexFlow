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


import gradio as gr
import flexflow.serve as ff
import argparse, json, os
from types import SimpleNamespace
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Any, List, Optional
import chainlit as cl

class FlexFlowLLM(LLM):
    def __init__(self, llm):
        self.llm = llm

    @property
    def _llm_type(self) -> str:
        return "flexflow"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        # 在这里调用 FlexFlow LLM 来生成回应
        result = self.llm.generate(prompt)
        return result.output_text.decode('utf-8')

    @property
    def _identifying_params(self) -> dict:
        # 返回识别参数，例如模型的名称
        return {"model": "FlexFlow Model"}
    

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
    


@cl.on_chat_start
def start_chat():
    global llm
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)

    ff.init(configs_dict)
    ff_data_type = ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    llm = ff.LLM(configs.llm_model, data_type=ff_data_type, cache_path=configs.cache_path, refresh_cache=configs.refresh_cache, output_file=configs.output_file)
    
    generation_config = ff.GenerationConfig(do_sample=False, temperature=0.9, topp=0.8, topk=1)
    llm.compile(generation_config, max_requests_per_batch=1, max_seq_length=256, max_tokens_per_batch=64)
    llm.start_server()
    
    flexflow_llm = FlexFlowLLM(llm)
    prompt_template = PromptTemplate(template="{query}", input_variables=['query'])
    llm_chain = LLMChain(llm=flexflow_llm, prompt=prompt_template, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def handle_message(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    text = message.content[:1200]
    res = await llm_chain.acall(text, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()

if __name__ == "__main__":
    cl.run()