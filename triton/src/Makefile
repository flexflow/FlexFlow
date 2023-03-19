#------------------------------------------------------------------------------#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif
ifndef TRITON_DIR
$(error TRITON_DIR variable is not defined, aborting build)
endif
ifndef PROTOBUF_DIR
$(error PROTOBUF_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
MAX_FIELDS	?= 256		# Maximum number of fields in a field space
OUTPUT_LEVEL    ?= LEVEL_INFO	# Compile time logging level
USE_FORTRAN	?= 0		# Include Fortran support		
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_OPENMP	?= 0		# Include OpenMP processor support
USE_NETWORK	?= 1		# Include support for multi-node execution
USE_ZLIB	?= 1		# Use ZLib for compression of log files
USE_LIBDL	?= 1		# Use LibDL for finding function pointer names
USE_LLVM	?= 0		# Include support for LLVM task variants
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
USE_SPY		?= 0		# Enable support for detailed Legion Spy logging
USE_HALF	?= 0		# Include support for half-precision reductions
USE_COMPLEX	?= 0		# Include support for complex type reductions
SHARED_OBJECTS	?= 0		# Generate shared objects for Legion and Realm
BOUNDS_CHECKS	?= 0		# Enable runtime bounds checks
PRIVILEGE_CHECKS ?= 0		# Enable runtime privilege checks
MARCH		?= native	# Set the name of the target CPU archiecture
GPU_ARCH	?= ampere	# Set the name of the target GPU architecture
CONDUIT		?= ibv		# Set the name of the GASNet conduit to use
REALM_NETWORKS	?= gasnetex	# Set the kind of networking layer to use
GASNET		?=		# Location of GASNet installation
CUDA		?=		# Location of CUDA installation
HDF_ROOT	?=		# Location of HDF5 installation
PREFIX		?= /usr		# Location of where to install Legion

# Put the binary file name here
OUTFILE		?= libtriton_legion.so
# List all the application source files here
CC_SRC		?=		# .c files
CXX_SRC		?= backend.cc \
		   model.cc \
		   runtime.cc \
		   instance.cc \
		   onnx_parser.cc \
		   onnx/onnx-data.pb.cc \
		   onnx/onnx-ml.pb.cc \
		   onnx/onnx-operators-ml.pb.cc \
		   operator.cc \
		   strategy.cc \
		   tensor.cc \
		   operators/unary.cc \
		   operators/pool2d.cc \
		   operators/concat.cc \
		   operators/conv2d.cc \
		   operators/matmul.cc \
		   operators/softmax.cc \
		   operators/reshape.cc # .cc files
CUDA_SRC	?= operators/unary.cu
#CUDA_SRC	?= flexflow/runtime/model.cu \
		   flexflow/ops/attention.cu \
		   flexflow/ops/batch_matmul.cu \
		   flexflow/ops/batch_norm.cu \
		   flexflow/ops/concat.cu \
		   flexflow/ops/conv_2d.cu \
		   flexflow/ops/dropout.cu \
		   flexflow/ops/element_binary.cu \
		   flexflow/ops/element_unary.cu \
		   flexflow/ops/embedding.cu \
		   flexflow/ops/flat.cu \
		   flexflow/ops/fused.cu \
		   flexflow/ops/linear.cu \
		   flexflow/ops/pool_2d.cu \
		   flexflow/ops/reshape.cu \
		   flexflow/ops/reverse.cu \
		   flexflow/ops/softmax.cu \
		   flexflow/ops/split.cu \
		   flexflow/ops/tanh.cu \
		   flexflow/ops/topk.cu \
		   flexflow/ops/transpose.cu # .cu files
FORT_SRC	?=		# .f90 files
ASM_SRC		?=		# .S files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I$(TRITON_DIR)/include -I$(PROTOBUF_DIR)/include # Include flags for all compilers
CC_FLAGS	?= -fPIC	# Flags for all C++ compilers
FC_FLAGS	?=		# Flags for all Fortran compilers
NVCC_FLAGS	?= -Xcompiler -fPIC	# Flags for all NVCC files
SO_FLAGS	?=		# Flags for building shared objects
LD_FLAGS	?= -shared -Wl,--version-script libtriton_legion.ldscript -L$(PROTOBUF_DIR)/lib -lprotobuf -L$(TRITON_DIR)/lib -ltritonasyncworkqueue -ltritonbackendutils -ltritonserver_stub # Flags for linking binaries
# Canonical GNU flags you can modify as well
CPPFLAGS 	?=
CFLAGS		?=
CXXFLAGS 	?=
FFLAGS 		?=
LDLIBS 		?=
LDFLAGS 	?=

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk

