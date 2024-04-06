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

from typing import Optional
from ..type import *
from flexflow.core import *
from .serve import LLM, SSM, GenerationConfig, GenerationResult


def __check_positive_int(configs_dict: dict, key: str):
    value = configs_dict.get(key, None)
    if value is not None:
        if type(value) is not int:
            raise TypeError(f"Parameter {key} has value {value}, which is not an int!")
        elif value <= 0:
            raise ValueError(
                f"Parameter {key} has value {value}, which is not a positive number!"
            )


def init(
    configs_dict: Optional[dict] = None,
    *,
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
    benchmarking: Optional[bool] = None,
    inference_debugging: Optional[bool] = None,
    fusion: Optional[bool] = None,
):
    """
    Configure FlexFlow Serve and start the runtime.

    The function takes, alternatively, configs_dict (a positional argument of type dictionary),
    or three mandatory named parameters, plus some additional optional named parameters. When passing
    a configs_dict, no named parameter should be specified, and the dictionary should have keys matching
    at least the mandatory named parameters.

    The three mandatory parameters, which cannot be changed after starting the runtime, are:
    - num_gpus: the number of GPUs to reserve for the runtime
    - memory_per_gpu: the amount of memory (in MB) to pre-allocate on each GPU
    - zero_copy_memory_per_node: the amount of zero-copy memory (in MB) to pre-allocate for each node

    The optional parameters are:
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
    - benchmarking: whether to run benchmaking only, without loading real weights, defaults to False
    - inference_debugging: whether to run inference in debugging mode, saving all inputs/outputs/weights to file, defaults to False
    - fusion: whether to enable the FlexFlow operator fusion optimization, defaults to True

    The configurations are passed down to the FlexFlow runtime (implemented in C++) via command line arguments.


    :param configs_dict: A Python dictionary to pass all configurations as a single object
    :type configs_dict: dict
    :param num_gpus: the number of GPUs to reserve for the runtime
    :type num_gpus: int
    :param memory_per_gpu: memory_per_gpu: the amount of memory (in MB) to pre-allocate on each GPU
    :type memory_per_gpu: int
    :param zero_copy_memory_per_node: zero_copy_memory_per_node: the amount of zero-copy memory (in MB) to pre-allocate for each node
    :type zero_copy_memory_per_node: int
    :param num_cpus: the number of CPU processors to reserve for the runtime, defaults to 4
    :type num_cpus: Optional[int], optional
    :param legion_utility_processors: number of Legion utility threads to create per process, defaults to 1
    :type legion_utility_processors: Optional[int], optional
    :param data_parallelism_degree: the degree of parallelization in the data parallel dimension, defaults to 1
    :type data_parallelism_degree: Optional[int], optional
    :param tensor_parallelism_degree: the degree of parallelization in the tensor parallel dimension (using the Megatron technique), defaults to 1
    :type tensor_parallelism_degree: Optional[int], optional
    :param pipeline_parallelism_degree: the degree of parallelization in the pipeline parallel dimension, defaults to 1
    :type pipeline_parallelism_degree: Optional[int], optional
    :param offload: whether to enable offloading of the weights to CPU, defaults to False
    :type offload: Optional[bool], optional
    :param offload_reserve_space_size: the space (in MB) to reserve on CPU for offloading, default to 1024^2
    :type offload_reserve_space_size: Optional[int], optional
    :param use_4bit_quantization: whether to use 4-bit quantization, defaults to False
    :type use_4bit_quantization: Optional[bool], optional
    :param use_8bit_quantization: whether to use 8-bit quantization, defaults to False
    :type use_8bit_quantization: Optional[bool], optional
    :param profiling: whether to enable the FlexFlow profiling mode, defaults to False
    :type profiling: Optional[bool], optional
    :param benchmarking: whether to run benchmaking only, without loading real weights, defaults to False
    :type benchmarking: Optional[bool], optional
    :param inference_debugging: whether to run inference in debugging mode, saving all inputs/outputs/weights to file, defaults to False
    :type inference_debugging: Optional[bool], optional
    :param fusion: whether to enable the FlexFlow operator fusion optimization, defaults to True
    :type fusion: Optional[bool], optional

    :raises ValueError: this function will raise an exception if the user passes both a configs_dict and some named parameters
    :raises TypeError: this function will raise an exception if the configs_dict is not a dictionary
    :raises ValueError: this function will raise an exception if the mandatory FlexFlow initialization parameters are missing, or are not positive integers: num_gpus, memory_per_gpu, zero_copy_memory_per_node
    """

    # Check that if configs_dict is passed, no other key-value argument (after the *) is passed.
    if configs_dict is not None and any(
        [
            num_gpus is not None,
            memory_per_gpu is not None,
            zero_copy_memory_per_node is not None,
            num_cpus is not None,
            legion_utility_processors is not None,
            data_parallelism_degree is not None,
            tensor_parallelism_degree is not None,
            pipeline_parallelism_degree is not None,
            offload is not None,
            offload_reserve_space_size is not None,
            use_4bit_quantization is not None,
            use_8bit_quantization is not None,
            profiling is not None,
            benchmarking is not None,
            inference_debugging is not None,
            fusion is not None,
        ]
    ):
        raise ValueError("Cannot pass both configs_dict and individual args")

    if configs_dict is not None:
        if type(configs_dict) != dict:
            raise TypeError("configs_dict is not a dictionary")
    else:
        # Add named key-value arguments into dictionary
        configs_dict = {
            "num_gpus": num_gpus,
            "memory_per_gpu": memory_per_gpu,
            "num_cpus": num_cpus,
            "zero_copy_memory_per_node": zero_copy_memory_per_node,
            "legion_utility_processors": legion_utility_processors,
            "data_parallelism_degree": data_parallelism_degree,
            "tensor_parallelism_degree": tensor_parallelism_degree,
            "pipeline_parallelism_degree": pipeline_parallelism_degree,
            "offload": offload,
            "offload_reserve_space_size": offload_reserve_space_size,
            "use_4bit_quantization": use_4bit_quantization,
            "use_8bit_quantization": use_8bit_quantization,
            "profiling": profiling,
            "benchmarking": benchmarking,
            "inference_debugging": inference_debugging,
            "fusion": fusion,
        }

    # Check that mandatory configs are present
    required_keys = ["num_gpus", "memory_per_gpu", "zero_copy_memory_per_node"]
    for required_key in required_keys:
        if configs_dict.get(required_key, None) is None:
            raise ValueError(
                "Missing one of the following required configs: num_gpus, memory_per_gpu, zero_copy_memory_per_node"
            )

    # Sanity check parameters
    positive_int_params = required_keys + [
        "legion_utility_processors",
        "data_parallelism_degree",
        "tensor_parallelism_degree",
        "pipeline_parallelism_degree",
        "offload_reserve_space_size",
    ]
    for param in positive_int_params:
        __check_positive_int(configs_dict, param)

    # Set default values
    if configs_dict.get("num_cpus", None) is None:
        configs_dict["num_cpus"] = 4
    if configs_dict.get("legion_utility_processors", None) is None:
        configs_dict["legion_utility_processors"] = 8
    if configs_dict.get("data_parallelism_degree", None) is None:
        configs_dict["data_parallelism_degree"] = 1
    if configs_dict.get("tensor_parallelism_degree", None) is None:
        configs_dict["tensor_parallelism_degree"] = 1
    if configs_dict.get("pipeline_parallelism_degree", None) is None:
        configs_dict["pipeline_parallelism_degree"] = 1
    if configs_dict.get("offload", None) is None:
        configs_dict["offload"] = False
    if configs_dict.get("offload_reserve_space_size", None) is None:
        configs_dict["offload_reserve_space_size"] = 1024**2
    if configs_dict.get("use_4bit_quantization", None) is None:
        configs_dict["use_4bit_quantization"] = False
    if configs_dict.get("use_8bit_quantization", None) is None:
        configs_dict["use_8bit_quantization"] = False
    if configs_dict.get("profiling", None) is None:
        configs_dict["profiling"] = False
    if configs_dict.get("benchmarking", None) is None:
        configs_dict["benchmarking"] = False
    if configs_dict.get("inference_debugging", None) is None:
        configs_dict["inference_debugging"] = False
    if configs_dict.get("fusion", None) is None:
        configs_dict["fusion"] = True

    init_flexflow_runtime(configs_dict)
