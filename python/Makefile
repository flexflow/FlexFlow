# Copyright 2020 Stanford University
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


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

ifndef FF_DIR
$(error FF_DIR variable is not defined, aborting build)
endif

ifndef PROTOBUF_DIR
PROTOBUF_DIR ?= /home/wwu12/opt
endif

ifndef CUDNN_DIR
CUDNN_DIR ?= /scratch/shared/wwu12/cudnn
endif

ifndef CUDA_DIR
CUDA_DIR ?= /spack/opt/spack/linux-scientific7-x86_64/gcc-7.3.0/cuda-10.0.130-etoqn3vlwx2os25n3rmtksntnu5ioorb
endif

PROTOBUF_EXE ?= $(PROTOBUF_DIR)/bin/protoc

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

USE_PYTHON := 1
PYTHON_EXE ?= python3.6
PYTHON_LIB ?= /usr/lib64/libpython3.6m.so.1.0
PYTHON_VERSION_MAJOR ?= 3

# Put the binary file name here
OUTFILE		?= flexflow_python

# List all the application source files here
FF_SRC ?= $(FF_DIR)/src/mapper/mapper.cc $(FF_DIR)/src/runtime/strategy.cc $(FF_DIR)/src/runtime/model.cc $(FF_DIR)/src/runtime/optimizer.cc $(FF_DIR)/src/runtime/initializer.cc $(FF_DIR)/src/ops/embedding.cc 
FF_GPU_SRC ?= $(FF_DIR)/src/ops/conv_2d.cu $(FF_DIR)/src/runtime/model.cu $(FF_DIR)/src/ops/pool_2d.cu $(FF_DIR)/src/ops/batch_norm.cu $(FF_DIR)/src/ops/linear.cu  \
		$(FF_DIR)/src/ops/softmax.cu $(FF_DIR)/src/ops/concat.cu $(FF_DIR)/src/ops/flat.cu $(FF_DIR)/src/ops/embedding.cu $(FF_DIR)/src/ops/mse_loss.cu\
		$(FF_DIR)/src/runtime/initializer_kernel.cu $(FF_DIR)/src/runtime/optimizer_kernel.cu $(FF_DIR)/src/runtime/accessor_kernel.cu\
		$(FF_DIR)/src/runtime/cuda_helper.cu
GEN_SRC		?= main.cc flexflow/hello/hello.cc flexflow_c.cc strategy.pb.cc $(FF_SRC) # .cc files
GEN_GPU_SRC	?= $(FF_GPU_SRC)		# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I$(FF_DIR)/include -I$(PROTOBUF_DIR)/include -I$(CUDNN_DIR)/include -I$(CUDA_DIR)/include 
CC_FLAGS	?= -DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON -std=c++11
NVCC_FLAGS	?= -std=c++11
GASNET_FLAGS	?=
LD_FLAGS	?= -L$(PROTOBUF_DIR)/lib -lprotobuf -L$(CUDNN_DIR)/lib64 -lcudnn -L$(CUDA_DIR)/lib64 -lcublas -lcurand

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -Wl,-force_load,liblegion.a
else
	LD_FLAGS += -Wl,--whole-archive -llegion -Wl,--no-whole-archive
endif

# Use reduction operators for complex types by default
CC_FLAGS += -DLEGION_REDOP_COMPLEX

NO_BUILD_ALL=1
.PHONY: all
all: $(OUTFILE) flexflow/core/legion_cffi.py strategy.pb.cc

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

flexflow/core/legion_cffi.py: legion_cffi.py.in legion_cffi_build.py
	$(PYTHON_EXE) legion_cffi_build.py --runtime-dir $(LG_RT_DIR) --defines-dir $(DEFINE_HEADERS_DIR) --output-dir flexflow/core
	
strategy.pb.cc: $(FF_DIR)/src/runtime/strategy.proto
	$(PROTOBUF_EXE) --proto_path=$(FF_DIR)/src/runtime strategy.proto --cpp_out=.

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(REALM_OBJS) $(LEGION_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) legion_defines.h realm_defines.h flexflow/core/legion_cffi.py *.pyc
	$(RM) -rf build dist *.egg-info
