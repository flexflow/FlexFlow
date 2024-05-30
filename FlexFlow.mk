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

ifndef HIPLIB_HOME
HIPLIB_HOME = /opt/rocm-4.3.1
endif

#ifndef MPI_HOME
#MPI_HOME = $(patsubst %/bin/mpicc,%,$(shell which mpicc | head -1))
#endif

ifeq ($(strip $(USE_GASNET)),1)
  ifndef GASNET
  $(error USE_GASNET is enabled, but GASNET variable is not defined, aborting build)
  endif
endif

# disable hijack
USE_CUDART_HIJACK = 1

GEN_SRC += $(shell find $(FF_HOME)/src/loss_functions/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/mapper/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/metrics_functions/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/ops/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/parallel_ops/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/recompile/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/runtime/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/utils/dot/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/dataloader/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/c/ -name '*.cc')\
		$(shell find $(FF_HOME)/inference/ -name 'file_loader.cc')
GEN_SRC := $(filter-out $(FF_HOME)/src/runtime/cpp_driver.cc, $(GEN_SRC))

FF_CUDA_SRC += $(shell find $(FF_HOME)/src/loss_functions/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/mapper/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/metrics_functions/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/ops/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/parallel_ops/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/recompile/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/runtime/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/utils/dot/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/dataloader/ -name '*.cu')

FF_HIP_SRC += $(shell find $(FF_HOME)/src/loss_functions/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/mapper/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/metrics_functions/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/ops/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/parallel_ops/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/recompile/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/runtime/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/utils/dot/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/dataloader/ -name '*.cpp')
		
GEN_GPU_SRC += $(FF_CUDA_SRC)
ifeq ($(strip $(HIP_TARGET)),CUDA)
  GEN_HIP_SRC += $(FF_CUDA_SRC)
else
  GEN_HIP_SRC += $(FF_HIP_SRC)
endif

ifneq ($(strip $(FF_USE_PYTHON)), 1)
  GEN_SRC		+= ${FF_HOME}/src/runtime/cpp_driver.cc
endif


INC_FLAGS	+= -I${FF_HOME}/include -I${FF_HOME}/inference -I${FF_HOME}/deps/optional/include -I${FF_HOME}/deps/variant/include -I${FF_HOME}/deps/json/include -I${FF_HOME}/deps/tokenizers-cpp/include -I${FF_HOME}/deps/tokenizers-cpp/sentencepiece/src
CC_FLAGS	+= -DMAX_TENSOR_DIM=$(MAX_DIM) -DLEGION_MAX_RETURN_SIZE=32768
NVCC_FLAGS	+= -DMAX_TENSOR_DIM=$(MAX_DIM) -DLEGION_MAX_RETURN_SIZE=32768
HIPCC_FLAGS     += -DMAX_TENSOR_DIM=$(MAX_DIM) -DLEGION_MAX_RETURN_SIZE=32768
GASNET_FLAGS	+=
# For Point and Rect typedefs
CC_FLAGS	+= -std=c++17
NVCC_FLAGS	+= -std=c++17
HIPCC_FLAGS     += -std=c++17

LD_FLAGS += -L$(FF_HOME)/deps/tokenizers-cpp/example/tokenizers -ltokenizers_cpp -ltokenizers_c -L$(FF_HOME)/deps/tokenizers-cpp/example/tokenizers/sentencepiece/src -lsentencepiece

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

ifeq ($(strip $(USE_CUDA)),1)
CC_FLAGS	+= -DFF_USE_CUDA
NVCC_FLAGS	+= -DFF_USE_CUDA
INC_FLAGS	+= -I$(CUDNN_HOME)/include -I$(CUDA_HOME)/include
LD_FLAGS	+= -lcudnn -lcublas -lcurand -L$(CUDNN_HOME)/lib64 -L$(CUDA_HOME)/lib64
endif

ifeq ($(strip $(USE_HIP)),1)
ifeq ($(strip $(HIP_TARGET)),CUDA)
CC_FLAGS	+= -DFF_USE_HIP_CUDA
HIPCC_FLAGS	+= -DFF_USE_HIP_CUDA
INC_FLAGS	+= -I$(CUDNN_HOME)/include -I$(CUDA_HOME)/include
LD_FLAGS	+= -lcudnn -lcublas -lcurand -L$(CUDNN_HOME)/lib64 -L$(CUDA_HOME)/lib64
else
CC_FLAGS	+= -DFF_USE_HIP_ROCM
HIPCC_FLAGS	+= -DFF_USE_HIP_ROCM
INC_FLAGS	+= -I$(HIPLIB_HOME)/include -I$(HIPLIB_HOME)/include/miopen -I$(HIPLIB_HOME)/include/rocrand -I$(HIPLIB_HOME)/include/hiprand 
LD_FLAGS	+= -lMIOpen -lhipblas -lhiprand -L$(HIPLIB_HOME)/lib
endif
endif

# CUDA arch variables
# We cannot use the default "auto" setting here because it will try to compile FlexFlow and Legion for GPU archs < 60, 
# which are not compatible with the half precision type.
GPU_ARCH ?= 60 61 62 70 72 75 80 90

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
