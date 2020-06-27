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
#$(error LG_RT_DIR variable is not defined, aborting build)
LG_RT_DIR	?= ../../legion/runtime
endif

ifndef FF_HOME
$(error FF_HOME variable is not defined, aborting build)
endif

ifndef CUDNN_HOME
$(error CUDNN_HOME variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= alexnet
# List all the application source files here
GEN_SRC		?= ../../src/runtime/model.cc ../../src/mapper/mapper.cc ../../src/runtime/initializer.cc ../../src/runtime/optimizer.cc\
		../../src/ops/embedding.cc ../../src/runtime/strategy.pb.cc ../../src/runtime/strategy.cc alexnet.cc
GEN_GPU_SRC	?= ../../src/ops/conv_2d.cu ../../src/runtime/model.cu ../../src/ops/pool_2d.cu ../../src/ops/batch_norm.cu ../../src/ops/linear.cu  \
		../../src/ops/softmax.cu ../../src/ops/concat.cu ../../src/ops/flat.cu ../../src/ops/embedding.cu ../../src/ops/mse_loss.cu\
		../../src/ops/element_binary.cu ../../src/ops/element_unary.cu\
		../../src/runtime/initializer_kernel.cu ../../src/runtime/optimizer_kernel.cu ../../src/runtime/accessor_kernel.cu\
		../../src/runtime/cuda_helper.cu# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I../../include/ -I${FF_HOME}/protobuf/src -I${CUDNN_HOME}/include
CC_FLAGS	?=
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?= -lcudnn -lcublas -lcurand -lprotobuf -L/usr/local/lib -L${FF_HOME}/protobuf/src/.libs -L${CUDNN_HOME}/lib64
# For Point and Rect typedefs
CC_FLAGS	+= -std=c++11
NVCC_FLAGS  	+= -std=c++11
###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

