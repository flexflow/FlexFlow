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
Running Instructions:
- To run this FastAPI application, make sure you have FastAPI and Uvicorn installed.
- Save this script as 'fastapi_specinfer.py'.
- Run the application using the command: `uvicorn fastapi_specinfer:app --reload --port PORT_NUMBER`
- The server will start on `http://localhost:PORT_NUMBER`. Use this base URL to make API requests.
- Go to `http://localhost:PORT_NUMBER/docs` for API documentation.
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import flexflow.serve as ff
import uvicorn
import json, os, argparse
from types import SimpleNamespace

# Initialize FastAPI application
app = FastAPI()

# Define the request model
class PromptRequest(BaseModel):
    prompt: str

# Global variable to store the LLM model
llm = None

def get_configs():
    # Fetch configuration file path from environment variable
    config_file = os.getenv("CONFIG_FILE", "")

    # Load configs from JSON file (if specified)
    if config_file:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        try:
            with open(config_file) as f:
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

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global llm

    # Initialize your LLM model configuration here
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)
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

# API endpoint to generate response
@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")
    
    # Call the model to generate a response
    full_output = llm.generate([prompt_request.prompt])[0].output_text.decode('utf-8')

    # Separate the prompt and response
    split_output = full_output.split('\n', 1)
    if len(split_output) > 1:
        response_text = split_output[1] 
    else:
        response_text = "" 
        
    # Return the prompt and the response in JSON format
    return {
        "prompt": prompt_request.prompt,
        "response": response_text
    }
    
# Shutdown event to stop the model server
@app.on_event("shutdown")
async def shutdown_event():
    global llm
    if llm is not None:
        llm.stop_server()

# Main function to run Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Running within the entrypoint folder:
# uvicorn fastapi_specinfer:app --reload --port

# Running within the python folder:
# uvicorn entrypoint.fastapi_specinfer:app --reload --port 3000
