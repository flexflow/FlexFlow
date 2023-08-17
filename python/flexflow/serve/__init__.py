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

import json, sys, os
from typing import Union, Optional
from ..type import *


def _parse_positive_int_config(name: str, variable: str, ff_cli_name: str = None):
    if variable is not None:
        if type(variable) is not int:
            raise ValueError(
                f"The following configs take positive integers only: {name}"
            )
        elif variable <= 0:
            raise ValueError(
                f"The following configs take positive integers only: {name}"
            )
        if not ff_cli_name:
            sys.argv += ["-{name}", str(variable)]
        else:
            sys.argv += [f"{ff_cli_name}", str(variable)]


def init(configs: Optional[str] = None,
        num_gpus: Optional[int] = None,
        memory_per_gpu: Optional[int] = None,
        zero_copy_memory_per_node: Optional[int] = None,
        num_cpus: Optional[int] = None,
        legion_utility_processors: Optional[int] = None,
        data_parallelism_degree: Optional[int] = None,
        tensor_parallelism_degree: Optional[int] = None,
        pipeline_parallelism_degree: Optional[int] = None,
        offload: Optional[bool] = None,
        offload_reserve_space_size: Optional[int] = None,
        use_4bit_quantization: Optional[bool] = None,
        use_8bit_quantization: Optional[bool] = None,
        profiling: Optional[bool] = None,
        fusion: Optional[bool] = None):
    """Configure FlexFlow for inference and start the FlexFlow runtime by importing the flexflow.core package.

    The configurations are passed down to the FlexFlow runtime (implemented in C++) via command line arguments.

    The init function takes three mandatory parameters, which cannot be changed after starting the runtime. These are:
    - num_gpus: the number of GPUs to reserve for the runtime
    - memory_per_gpu: the amount of memory (in MB) to pre-allocate on each GPU
    - zero_copy_memory_per_node: the amount of zero-copy memory (in MB) to pre-allocate for each node

    In addition, the following optional parameters can be passed:
    - num_cpus: the number of CPU processors to reserve for the runtime, defaults to 4
    - legion_utility_processors: number of Legion utility threads to create per process, defaults to 1
    - data_parallelism_degree: the degree of parallelization in the data parallel dimension, defaults to 1
    - tensor_parallelism_degree: the degree of parallelization in the tensor parallel dimension (using the Megatron technique), defaults to 1
    - pipeline_parallelism_degree: the degree of parallelization in the pipeline parallel dimension, defaults to 1
    - offload: whether to enable offloading of the weights to CPU, defaults to False
    - offload_reserve_space_size: the space (in MB) to reserve on CPU for offloading, default to 1024^2
    - use_4bit_quantization: whether to use 4-bit quantization, defaults to False
    - use_8bit_quantization: whether to use 8-bit quantization, defaults to False
    - profiling: whether to enable the FlexFlow profiling mode, defaults to False
    - fusion: whether to enable the FlexFlow operator fusion optimization, defaults to True

    :param configs: The runtime configs, in the form of the path to a JSON file
    :type configs: Union[str, dict]
    :raises ValueError: This function will raise an exception if the JSON file pointed to by the input string is not in the right format
    :raises ValueError: This function will raise an exception if the mandatory FlexFlow initialization parameters are missing, or are not positive integers: num_gpus, memory_per_gpu, zero_copy_memory_per_node
    """
    configs_dict = {}
    
    if configs is not None:
      if type(configs) == str:
        try:
            with open(configs) as f:
                configs_dict = json.load(f)
        except json.JSONDecodeError as e:
            print("JSON format error:")
            print(e)
      else:
        raise ValueError(
            "configs should be a valid JSON file"
        )
        
    # Remove the arguments to avoid interferences
    sys.argv = [sys.argv[0]]
    
    if configs is not None:
        # configs should contain the following mandatory keys with non-zero integer values:
        num_gpus = configs_dict.get("num_gpus")
        memory_per_gpu = configs_dict.get("memory_per_gpu")
        zero_copy_memory_per_node = configs_dict.get("zero_copy_memory_per_node")
        if not num_gpus or not memory_per_gpu or not zero_copy_memory_per_node:
            raise ValueError(
                "Missing one of the following configs in config dict: num_gpus, memory_per_gpu, zero_copy_memory_per_node"
            )
        num_cpus = configs_dict.get("num_cpus")
        legion_utility_processors = configs_dict.get("legion_utility_processors")
        data_parallelism_degree = configs_dict.get("data_parallelism_degree")
        tensor_parallelism_degree = configs_dict.get("tensor_parallelism_degree")
        pipeline_parallelism_degree = configs_dict.get("pipeline_parallelism_degree")
        offload = configs_dict.get("offload", False)
        offload_reserve_space_size = configs_dict.get("offload_reserve_space_size")
        use_4bit_quantization = configs_dict.get("use_4bit_quantization", False)
        use_8bit_quantization = configs_dict.get("use_8bit_quantization", False)
        profiling = configs_dict.get("profiling", False)
        fusion = configs_dict.get("fusion", True)
    else:
        if not num_gpus or not memory_per_gpu or not zero_copy_memory_per_node:
            raise ValueError(
            "Missing one of the following configs in input params: num_gpus, memory_per_gpu, zero_copy_memory_per_node"
        )
        offload = False if offload is None else offload
        use_4bit_quantization = False if use_4bit_quantization is None else use_4bit_quantization
        use_8bit_quantization = False if use_8bit_quantization is None else use_8bit_quantization
        profiling = False if profiling is None else profiling
        fusion = True if fusion is None else fusion
        
               
    # parse arguments     
    _parse_positive_int_config("num_gpus", num_gpus, "-ll:gpu")
    _parse_positive_int_config("memory_per_gpu", memory_per_gpu, "-ll:fsize")
    _parse_positive_int_config(
        "zero_copy_memory_per_node", zero_copy_memory_per_node, "-ll:zsize"
    )

    # parse optional arguments
    _parse_positive_int_config("num_cpus", num_cpus, "-ll:cpu")
    _parse_positive_int_config(
        "legion_utility_processors", legion_utility_processors, "-ll:util"
    )
    _parse_positive_int_config(
        "data_parallelism_degree", data_parallelism_degree, "-data-parallelism-degree"
    )
    _parse_positive_int_config(
        "tensor_parallelism_degree",
        tensor_parallelism_degree,
        "-tensor-parallelism-degree",
    )
    _parse_positive_int_config(
        "pipeline_parallelism_degree",
        pipeline_parallelism_degree,
        "-pipeline-parallelism-degree",
    )
    if offload:
        sys.argv += ["-offload"]
    _parse_positive_int_config(
        "offload_reserve_space_size",
        offload_reserve_space_size,
        "-offload-reserve-space-size",
    )
    if use_4bit_quantization:
        sys.argv += ["--4bit-quantization"]
    if use_8bit_quantization:
        sys.argv += ["--8bit-quantization"]
    if profiling:
        sys.argv += ["--profiling"]
    if fusion:
        sys.argv += ["--fusion"]

    global LLM, SSM, GenerationConfig
    from .serve import LLM, SSM, GenerationConfig


def init_cpu():
    """Start the FlexFlow runtime and import the inference package without access to GPU functionalities.
    This is useful to access the utilies from the flexflow package without using up GPU memory.
    """
    # Remove the arguments to avoid interferences
    sys.argv = [sys.argv[0]]
    # Ask the runtime to avoid using GPU/GPU memory
    os.environ["CPU_ONLY_TEST"] = "1"

    global LLM, SSM, GenerationConfig
    from .serve import LLM, SSM, GenerationConfig
