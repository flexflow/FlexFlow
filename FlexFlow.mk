# Copyright 2021 Stanford, Facebook, LANL
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

ifndef FF_HOME
$(error FF_HOME variable is not defined, aborting build)
endif

ifndef LG_RT_DIR
LG_RT_DIR	?= $(FF_HOME)/deps/legion/runtime
endif

ifndef CUDA_HOME
CUDA_HOME = $(patsubst %/bin/nvcc,%,$(shell which nvcc | head -1))
endif

ifndef CUDNN_HOME
CUDNN_HOME = $(CUDA_HOME)
endif

ifndef NCCL_HOME
NCCL_HOME = $(CUDA_HOME)
endif

#ifndef MPI_HOME
#MPI_HOME = $(patsubst %/bin/mpicc,%,$(shell which mpicc | head -1))
#endif

ifeq ($(strip $(USE_GASNET)),1)
  ifndef GASNET
  $(error USE_GASNET is enabled, but GASNET variable is not defined, aborting build)
  endif
endif

GEN_SRC		+= ${FF_HOME}/src/runtime/model.cc\
		${FF_HOME}/src/mapper/mapper.cc\
		${FF_HOME}/src/runtime/initializer.cc\
		${FF_HOME}/src/runtime/optimizer.cc\
		${FF_HOME}/src/ops/embedding.cc\
		${FF_HOME}/src/runtime/strategy.cc\
		${FF_HOME}/src/runtime/simulator.cc\
		${FF_HOME}/src/metrics_functions/metrics_functions.cc\
		${FF_HOME}/src/recompile/recompile_state.cc\
		${FF_HOME}/src/runtime/machine_model.cc

FF_CUDA_SRC	+= ${FF_HOME}/src/ops/conv_2d.cu\
		${FF_HOME}/src/runtime/model.cu\
		${FF_HOME}/src/ops/pool_2d.cu\
		${FF_HOME}/src/ops/batch_norm.cu\
		${FF_HOME}/src/ops/linear.cu\
		${FF_HOME}/src/ops/softmax.cu\
		${FF_HOME}/src/ops/concat.cu\
		${FF_HOME}/src/ops/split.cu\
		${FF_HOME}/src/ops/dropout.cu\
		${FF_HOME}/src/ops/flat.cu\
		${FF_HOME}/src/ops/embedding.cu\
		${FF_HOME}/src/ops/element_binary.cu\
		${FF_HOME}/src/ops/element_unary.cu\
		${FF_HOME}/src/ops/batch_matmul.cu\
		${FF_HOME}/src/ops/reshape.cu\
		${FF_HOME}/src/ops/reverse.cu\
		${FF_HOME}/src/ops/topk.cu\
		${FF_HOME}/src/ops/cache.cu\
		${FF_HOME}/src/ops/group_by.cu\
		${FF_HOME}/src/ops/aggregate.cu\
		${FF_HOME}/src/ops/aggregate_spec.cu\
		${FF_HOME}/src/ops/transpose.cu\
		${FF_HOME}/src/ops/attention.cu\
		${FF_HOME}/src/ops/fused.cu\
		${FF_HOME}/src/loss_functions/loss_functions.cu\
		${FF_HOME}/src/metrics_functions/metrics_functions.cu\
		${FF_HOME}/src/runtime/initializer_kernel.cu\
		${FF_HOME}/src/runtime/optimizer_kernel.cu\
		${FF_HOME}/src/runtime/accessor_kernel.cu\
		${FF_HOME}/src/runtime/simulator.cu\
		${FF_HOME}/src/runtime/cuda_helper.cu
		
GEN_GPU_SRC += $(FF_CUDA_SRC)
GEN_HIP_SRC += $(FF_CUDA_SRC)

ifneq ($(strip $(FF_USE_PYTHON)), 1)
  GEN_SRC		+= ${FF_HOME}/src/runtime/cpp_driver.cc
endif

INC_FLAGS	+= -I${FF_HOME}/include/ -I$(CUDNN_HOME)/include -I$(CUDA_HOME)/include
LD_FLAGS	+= -lcudnn -lcublas -lcurand -L$(CUDNN_HOME)/lib64 -L$(CUDA_HOME)/lib64
CC_FLAGS	+= -DMAX_TENSOR_DIM=$(MAX_DIM)
NVCC_FLAGS	+= -DMAX_TENSOR_DIM=$(MAX_DIM)
HIPCC_FLAGS     += -DMAX_TENSOR_DIM=$(MAX_DIM)
GASNET_FLAGS	+=
# For Point and Rect typedefs
CC_FLAGS	+= -std=c++11
NVCC_FLAGS	+= -std=c++11
HIPCC_FLAGS     += -std=c++11

ifeq ($(strip $(FF_USE_NCCL)), 1)
INC_FLAGS	+= -I$(MPI_HOME)/include -I$(NCCL_HOME)/include
CC_FLAGS	+= -DFF_USE_NCCL
NVCC_FLAGS	+= -DFF_USE_NCCL
HIPCC_FLAGS     += -DFF_USE_NCCL
LD_FLAGS	+= -L$(NCCL_HOME)/lib -lnccl
endif

ifeq ($(strip $(FF_USE_AVX2)), 1)
CC_FLAGS	+= -DFF_USE_AVX2 -mavx2
endif

#ifndef HDF5
#HDF5_inc	?= /usr/include/hdf5/serial
#HDF5_lib	?= /usr/lib/x86_64-linux-gnu/hdf5/serial
#INC_FLAGS	+= -I${HDF5}/
#LD_FLAGS	+= -L${HDF5_lib} -lhdf5
#endif

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk
