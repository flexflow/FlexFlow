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

PROTOBUF_DIR ?= /home/wwu12/opt
ifndef PROTOBUF_DIR
$(error PROTOBUF_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
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
GEN_SRC		?= main.cc hello/hello.cc $(FF_DIR)/src/mapper/mapper.cc $(FF_DIR)/src/runtime/strategy.cc $(FF_DIR)/src/runtime/strategy.pb.cc # .cc files
GEN_GPU_SRC	?=		# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I$(FF_DIR)/include -I$(PROTOBUF_DIR)/include
CC_FLAGS	?= -DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_CPU
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?= -L$(PROTOBUF_DIR)/lib -lprotobuf

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -Wl,-force_load,liblegion.a
else
	LD_FLAGS += -Wl,--whole-archive -llegion -Wl,--no-whole-archive
endif

# Use reduction operators for complex types by default
CC_FLAGS += -DLEGION_REDOP_COMPLEX

NO_BUILD_ALL=1
.PHONY: all
all: $(OUTFILE) legion_cffi.py

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk

legion_cffi.py: legion_cffi.py.in legion_cffi_build.py
	$(PYTHON_EXE) legion_cffi_build.py --defines-dir $(DEFINE_HEADERS_DIR)

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(REALM_OBJS) $(LEGION_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) legion_defines.h realm_defines.h legion_cffi.py *.pyc
	$(RM) -rf build dist *.egg-info
