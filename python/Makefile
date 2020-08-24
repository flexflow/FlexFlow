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

ifndef FF_HOME
$(error FF_HOME variable is not defined, aborting build)
endif

#ifndef CUDA_HOME
#$(error CUDA_HOME variable is not defined, aborting build)
#endif

#ifndef CUDNN_HOME
#$(error CUDNN_HOME variable is not defined, aborting build)
#endif

ifndef LG_RT_DIR
LG_RT_DIR	?= $(FF_HOME)/legion/runtime
endif

ifdef PROTOBUF_DIR
PROTOBUF_EXE ?= $(PROTOBUF_DIR)/bin/protoc
PROTOBUF_LIB ?= $(PROTOBUF_DIR)/lib
PROTOBUF_INC ?= $(PROTOBUF_DIR)/include
else
PROTOBUF_EXE ?= $(FF_HOME)/protobuf/src/protoc
PROTOBUF_LIB ?= $(FF_HOME)/protobuf/src/.libs
PROTOBUF_INC ?= ${FF_HOME}/protobuf/src
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

USE_PYTHON := 1
ifndef PYTHON_LIB
  ifndef PYTHON_ROOT
    PYTHON_EXE := $(shell which python3 | head -1)
    ifeq ($(PYTHON_EXE),)
      $(error cannot find python - set PYTHON_ROOT if not in PATH)
    endif
    PYTHON_VERSION_MAJOR := $(shell $(PYTHON_EXE) -c 'import sys; print(sys.version_info.major)')
    PYTHON_VERSION_MINOR := $(shell $(PYTHON_EXE) -c 'import sys; print(sys.version_info.minor)')
    PYTHON_ROOT := $(dir $(PYTHON_EXE))
  endif

  # Try searching for common locations of the Python shared library.
  ifneq ($(strip $(PYTHON_ROOT)),)
    ifeq ($(strip $(DARWIN)),1)
      PYTHON_EXT := dylib
    else
      PYTHON_EXT := so
    endif
    PYTHON_LIB := $(wildcard $(PYTHON_ROOT)/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT))
    ifeq ($(strip $(PYTHON_LIB)),)
      PYTHON_LIB := $(wildcard $(abspath $(PYTHON_ROOT)/../lib/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT)))
      ifeq ($(strip $(PYTHON_LIB)),)
        PYTHON_LIB := $(wildcard $(abspath $(PYTHON_ROOT)/../lib64/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT)))
        ifeq ($(strip $(PYTHON_LIB)),)
          $(warning cannot find libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT) - falling back to using LD_LIBRARY_PATH)
          PYTHON_LIB :=
        endif
      endif
    endif
  endif
endif

# Put the binary file name here
OUTFILE		?= flexflow_python

# List all the application source files here
FF_SRC ?=	$(FF_HOME)/src/mapper/mapper.cc $(FF_HOME)/src/runtime/strategy.cc $(FF_HOME)/src/metrics_functions/metrics_functions.cc \
		$(FF_HOME)/src/runtime/model.cc $(FF_HOME)/src/runtime/optimizer.cc $(FF_HOME)/src/runtime/initializer.cc \
		$(FF_HOME)/src/runtime/simulator.cc $(FF_HOME)/src/ops/embedding.cc 
FF_GPU_SRC ?= 	$(FF_HOME)/src/ops/conv_2d.cu $(FF_HOME)/src/runtime/model.cu\
		$(FF_HOME)/src/ops/pool_2d.cu $(FF_HOME)/src/ops/batch_norm.cu\
		$(FF_HOME)/src/ops/linear.cu  $(FF_HOME)/src/ops/dropout.cu\
		$(FF_HOME)/src/ops/softmax.cu $(FF_HOME)/src/ops/concat.cu\
		$(FF_HOME)/src/ops/flat.cu $(FF_HOME)/src/ops/embedding.cu \
		$(FF_HOME)/src/ops/element_binary.cu $(FF_HOME)/src/ops/element_unary.cu \
    		$(FF_HOME)/src/loss_functions/loss_functions.cu\
		$(FF_HOME)/src/metrics_functions/metrics_functions.cu\
		$(FF_HOME)/src/runtime/initializer_kernel.cu $(FF_HOME)/src/runtime/optimizer_kernel.cu\
		$(FF_HOME)/src/runtime/accessor_kernel.cu $(FF_HOME)/src/runtime/simulator.cu\
		$(FF_HOME)/src/runtime/cuda_helper.cu flexflow_dataloader.cu
GEN_SRC		?= main.cc flexflow_c.cc flexflow_dataloader.cc strategy.pb.cc $(FF_SRC) # .cc files
GEN_GPU_SRC	?= $(FF_GPU_SRC) # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I$(FF_HOME)/include -I$(PROTOBUF_INC) -I$(CUDNN_HOME)/include -I$(CUDA_HOME)/include
CC_FLAGS	?= -DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON -std=c++11
NVCC_FLAGS	?= -std=c++11
GASNET_FLAGS	?=
LD_FLAGS	?= -lprotobuf -lcudnn -lcublas -lcurand -L$(PROTOBUF_LIB) -L$(CUDNN_HOME)/lib64 -L$(CUDA_HOME)/lib64

ifeq ($(strip $(FF_ENABLE_DEBUG)), 1)
CC_FLAGS += -DFF_DEBUG
NVCC_FLAGS += -DFF_DEBUG
endif

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -Wl,-force_load,liblegion.a
else
	LD_FLAGS += -Wl,--whole-archive -llegion -Wl,--no-whole-archive
endif

# Use reduction operators for complex types by default
CC_FLAGS += -DLEGION_REDOP_COMPLEX

NO_BUILD_ALL=1
.PHONY: all
all: strategy.pb.cc $(OUTFILE) flexflow/core/legion_cffi.py 

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

DEFINE_HEADERS_DIR ?= ./

flexflow/core/legion_cffi.py: legion_cffi.py.in legion_cffi_build.py
	$(PYTHON_EXE) legion_cffi_build.py --runtime-dir $(LG_RT_DIR) --defines-dir $(DEFINE_HEADERS_DIR) --output-dir flexflow/core
	
strategy.pb.cc: $(FF_HOME)/src/runtime/strategy.proto
	$(PROTOBUF_EXE) --proto_path=$(FF_HOME)/src/runtime strategy.proto --cpp_out=.

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(REALM_OBJS) $(LEGION_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) legion_defines.h realm_defines.h flexflow/core/legion_cffi.py *.pyc
	$(RM) -rf build dist *.egg-info
