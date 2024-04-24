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
#

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import atexit
import os
import sys
import warnings
from typing import Optional

from flexflow.config import *

# check which python binding to use
if flexflow_python_binding() == "pybind11":
    # print("Using pybind11 flexflow bindings.")
    from .flexflow_pybind11 import *
else:
    # print("Using cffi flexflow bindings.")
    from .flexflow_cffi import *

ff_arg_to_sysarg = {
    # General args
    "num_gpus": "-ll:gpu",
    "memory_per_gpu": "-ll:fsize",
    "zero_copy_memory_per_node": "-ll:zsize",
    "num_cpus": "-ll:cpu",
    "legion_utility_processors": "-ll:util",
    "profiling": "--profiling",
    "benchmarking": "--benchmarking",
    "inference_debugging": "--inference-debugging",
    "fusion": "--fusion",
    "disable_control_replication": "--disable-control-replication",
    # Training args
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "learning_rate": "--learning-rate",
    "weight_decay": "--weight-decay",
    "print_frequency": "--print-freq",
    "dataset": "--dataset",
    "budget": "--budget",
    "search_budget": "--search-budget",
    "alpha": "--alpha",
    "search_alpha": "--search-alpha",
    "simulator_workspace_size": "--simulator-workspace-size",
    "import": "--import",
    "import_strategy": "--import-strategy",
    "export": "--export",
    "export_strategy": "--export-strategy",
    "only_data_parallel": "--only-data-parallel",
    "enable_parameter_parallel": "--enable-parameter-parallel",
    "enable_attribute_parallel": "--enable-attribute-parallel",
    "allow_tensor_op_math_conversion": "--allow-tensor-op-math-conversion",
    "search_overlap_backward_update": "--overlap",
    "export_strategy_task_graph_file": "--taskgraph",
    "include_costs_dot_graph": "--include-costs-dot-graph",
    "export_strategy_computation_graph_file": "--compgraph",
    "machine_model_version": "--machine-model-version",
    "machine_model_file": "--machine-model-file",
    "simulator_segment_size": "--simulator-segment-size",
    "simulator_max_num_segments": "--simulator-max-num-segments",
    "enable_propagation": "--enable-propagation",
    "enable_inplace_optimizations": "--enable-inplace-optimization",
    "search_num_nodes": "--search-num-nodes",
    "search_num_workers": "--search-num-workers",
    "base_optimize_threshold": "--base-optimize-threshold",
    "python_data_loader_type": "--python-data-loader-type",
    "substitution_json_path": "--substitution-json",
    "perform_memory_search": "--memory-search",
    # Inference args
    "data_parallelism_degree": "-data-parallelism-degree",
    "tensor_parallelism_degree": "-tensor-parallelism-degree",
    "pipeline_parallelism_degree": "-pipeline-parallelism-degree",
    "offload": "-offload",
    "offload_reserve_space_size": "-offload-reserve-space-size",
    "use_4bit_quantization": "--4bit-quantization",
    "use_8bit_quantization": "--8bit-quantization"
}


def init_flexflow_runtime(configs_dict: Optional[dict] = None, **kwargs):
    if not flexflow_already_initialized():
        os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
        from legion_cffi import is_legion_python
        from .flexflowlib import flexflow_library

        # Default python mode
        if is_legion_python == False:
            # print("Using Default Python")
            from legion_top import (
                legion_canonical_python_main,
                legion_canonical_python_cleanup,
            )

            # Either a configs_dict dictionary, or individual key-value parameters should be passed. Not both.
            if configs_dict is not None and len(kwargs.items()) > 0:
                raise ValueError("Cannot pass both configs_dict and individual args")
            ff_args = configs_dict if configs_dict is not None else dict(kwargs.items())
            # Check presence of mandatory parameters
            if (
                "num_gpus" not in ff_args
                or "memory_per_gpu" not in ff_args
                or "zero_copy_memory_per_node" not in ff_args
            ):
                raise ValueError(
                    "Missing one of the following required configs: num_gpus, memory_per_gpu, zero_copy_memory_per_node"
                )

            # Remove any existing arguments to avoid interferences
            sys.argv = [sys.argv[0]]

            # Pass parameters to the FlexFlow C++ runtime via command line arguments
            for arg in ff_args:
                if arg not in ff_arg_to_sysarg:
                    # warnings.warn(f"Ignoring parameter {arg}: not recognized.")
                    continue
                else:
                    sys_arg = [ff_arg_to_sysarg[arg]]
                    if type(ff_args[arg]) == bool:
                        if ff_args[arg] is not True:
                            continue
                    else:
                        sys_arg += [str(ff_args[arg])]
                    sys.argv += sys_arg

            legion_canonical_python_main(sys.argv)
            atexit.register(legion_canonical_python_cleanup)
        else:
            # print("Using FlexFlow Python")
            if configs_dict is not None or len(kwargs.items()) > 0:
                warnings.warn("init_flexflow_runtime are ignored when using the FlexFlow Python interpreter")

        flexflow_library.initialize()
        set_flexflow_initialized()
    else:
        warnings.warn("Attempting to initialize FlexFlow more than once")
