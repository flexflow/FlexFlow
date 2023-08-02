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

import json, sys
from typing import Union
from ..type import *


def parse_positive_int_config(name: str, variable: str, ff_cli_name: str = None):
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


def init(configs: Union[str, dict]):
    """Configure FlexFlow for inference and start the FlexFlow runtime by importing the flexflow.core package.

    The init function takes three mandatory parameters, which cannot be changed after starting the runtime. These are:
    - num_gpus: the number of GPUs to reserve for the runtime
    - memory_per_gpu: the amount of memory (in MB) to pre-allocate on each GPU
    - zero_copy_memory_per_gpu: the amount of zero-copy memory (in MB) to pre-allocate for each GPU

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

    :param configs: The runtime configs, in the form of a dictionary or the path to a JSON file
    :type configs: Union[str, dict]
    :raises ValueError: This function will raise an exception if the JSON file pointed to by the input string is not in the right format
    :raises ValueError: This function will raise an exception if the mandatory FlexFlow initialization parameters are missing, or are not positive integers: num_gpus, memory_per_gpu, zero_copy_memory_per_gpu
    """
    configs_dict = {}
    if type(configs) == str:
        try:
            with open(configs) as f:
                configs_dict = json.load(f)
        except json.JSONDecodeError as e:
            print("JSON format error:")
            print(e)
    elif type(configs) == dict:
        configs_dict = configs
    else:
        raise ValueError(
            "configs should be a dictionary or the path to a valid JSON file"
        )

    # configs should contain the following mandatory keys with non-zero integer values:
    num_gpus = configs_dict.get("num_gpus")
    memory_per_gpu = configs_dict.get("memory_per_gpu")
    zero_copy_memory_per_gpu = configs_dict.get("zero_copy_memory_per_gpu")
    if not num_gpus or not memory_per_gpu or not zero_copy_memory_per_gpu:
        raise ValueError(
            "Missing one of the following configs: num_gpus, memory_per_gpu, zero_copy_memory_per_gpu"
        )
    parse_positive_int_config("num_gpus", num_gpus, "-ll:gpu")
    parse_positive_int_config("memory_per_gpu", memory_per_gpu, "-ll:fsize")
    parse_positive_int_config(
        "zero_copy_memory_per_gpu", zero_copy_memory_per_gpu, "-ll:zsize"
    )

    # parse optional arguments
    num_cpus = configs_dict.get("num_cpus")
    parse_positive_int_config("num_cpus", num_cpus, "-ll:cpu")
    legion_utility_processors = configs_dict.get("legion_utility_processors")
    parse_positive_int_config(
        "legion_utility_processors", legion_utility_processors, "-ll:util"
    )

    data_parallelism_degree = configs_dict.get("data_parallelism_degree")
    tensor_parallelism_degree = configs_dict.get("tensor_parallelism_degree")
    pipeline_parallelism_degree = configs_dict.get("pipeline_parallelism_degree")
    parse_positive_int_config(
        "data_parallelism_degree", data_parallelism_degree, "-data-parallelism-degree"
    )
    parse_positive_int_config(
        "tensor_parallelism_degree",
        tensor_parallelism_degree,
        "-tensor-parallelism-degree",
    )
    parse_positive_int_config(
        "pipeline_parallelism_degree",
        pipeline_parallelism_degree,
        "-pipeline-parallelism-degree",
    )

    offload = configs_dict.get("offload", False)
    if offload:
        sys.argv += ["-offload"]
    offload_reserve_space_size = configs_dict.get("offload_reserve_space_size")
    parse_positive_int_config(
        "offload_reserve_space_size",
        offload_reserve_space_size,
        "-offload-reserve-space-size",
    )
    use_4bit_quantization = configs_dict.get("use_4bit_quantization", False)
    if use_4bit_quantization:
        sys.argv += ["--4bit-quantization"]
    use_8bit_quantization = configs_dict.get("use_8bit_quantization", False)
    if use_8bit_quantization:
        sys.argv += ["--8bit-quantization"]

    profiling = configs_dict.get("profiling", False)
    if profiling:
        sys.argv += ["--profiling"]
    fusion = configs_dict.get("fusion", True)
    if fusion:
        sys.argv += ["--fusion"]

    global LLM, SSM, SamplingConfig
    from .serve import LLM, SSM, SamplingConfig
