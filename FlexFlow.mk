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

GEN_SRC += $(shell find $(FF_HOME)/src/loss_functions/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/mapper/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/metrics_functions/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/ops/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/parallel_ops/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/recompile/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/runtime/ -name '*.cc')\
		$(shell find $(FF_HOME)/src/utils/dot/ -name '*.cc')
GEN_SRC := $(filter-out $(FF_HOME)/src/runtime/cpp_driver.cc, $(GEN_SRC))

FF_CUDA_SRC += $(shell find $(FF_HOME)/src/loss_functions/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/mapper/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/metrics_functions/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/ops/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/parallel_ops/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/recompile/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/runtime/ -name '*.cu')\
		$(shell find $(FF_HOME)/src/utils/dot/ -name '*.cu')

FF_HIP_SRC += $(shell find $(FF_HOME)/src/loss_functions/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/mapper/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/metrics_functions/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/ops/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/parallel_ops/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/recompile/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/runtime/ -name '*.cpp')\
		$(shell find $(FF_HOME)/src/utils/dot/ -name '*.cpp')
		
GEN_GPU_SRC += $(FF_CUDA_SRC)
ifeq ($(strip $(HIP_TARGET)),CUDA)
  GEN_HIP_SRC += $(FF_CUDA_SRC)
else
  GEN_HIP_SRC += $(FF_HIP_SRC)
endif

ifneq ($(strip $(FF_USE_PYTHON)), 1)
  GEN_SRC		+= ${FF_HOME}/src/runtime/cpp_driver.cc
endif


INC_FLAGS	+= -I${FF_HOME}/include -I${FF_HOME}/deps/optional/include -I${FF_HOME}/deps/variant/include -I${FF_HOME}/deps/json/include -I${FF_HOME}/deps/sentencepiece/src
CC_FLAGS	+= -DMAX_TENSOR_DIM=$(MAX_DIM) -DLEGION_MAX_RETURN_SIZE=32768
NVCC_FLAGS	+= -DMAX_TENSOR_DIM=$(MAX_DIM) -DLEGION_MAX_RETURN_SIZE=32768
HIPCC_FLAGS     += -DMAX_TENSOR_DIM=$(MAX_DIM) -DLEGION_MAX_RETURN_SIZE=32768
GASNET_FLAGS	+=
# For Point and Rect typedefs
CC_FLAGS	+= -std=c++17
NVCC_FLAGS	+= -std=c++17
HIPCC_FLAGS     += -std=c++17

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
GPU_ARCH ?= all

# translate legacy arch names into numbers
ifeq ($(strip $(GPU_ARCH)),pascal)
override GPU_ARCH = 60
NVCC_FLAGS	+= -DPASCAL_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),volta)
override GPU_ARCH = 70
NVCC_FLAGS	+= -DVOLTA_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),turing)
override GPU_ARCH = 75
NVCC_FLAGS	+= -DTURING_ARCH
endif
ifeq ($(strip $(GPU_ARCH)),ampere)
override GPU_ARCH = 80
NVCC_FLAGS	+= -DAMPERE_ARCH
endif

ifeq ($(strip $(GPU_ARCH)),all)
  # detect based on what nvcc supports
  ALL_ARCHES = 60 61 62 70 72 75 80 86
  override GPU_ARCH = $(shell for X in $(ALL_ARCHES) ; do \
    $(NVCC) -gencode arch=compute_$$X,code=sm_$$X -cuda -x c++ /dev/null -o /dev/null 2> /dev/null && echo $$X; \
  done)
endif

# finally, convert space-or-comma separated list of architectures (e.g. 35,50)
#  into nvcc -gencode arguments
ifeq ($(findstring nvc++,$(shell $(NVCC) --version)),nvc++)
NVCC_FLAGS += $(foreach X,$(subst $(COMMA), ,$(GPU_ARCH)),-gpu=cc$(X))
else
COMMA=,
NVCC_FLAGS += $(foreach X,$(subst $(COMMA), ,$(GPU_ARCH)),-gencode arch=compute_$(X)$(COMMA)code=sm_$(X))
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
