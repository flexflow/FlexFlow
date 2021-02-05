# Copyright 2021 Stanford, LANL, Facebook
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

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 4		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)
#FF_ENABLE_NCCL  ?= 1
#USE_CUDART_HIJACK ?= 0

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
OUTFILE		  ?= flexflow_python

GEN_SRC		  ?= main.cc flexflow_c.cc flexflow_dataloader.cc # .cc files
GEN_GPU_SRC	  ?= flexflow_dataloader.cu # .cu files

CC_FLAGS	  ?= -DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON 

ifeq ($(strip $(FF_ENABLE_DEBUG)), 1)
CC_FLAGS += -DFF_DEBUG
NVCC_FLAGS += -DFF_DEBUG
endif

ifeq ($(strip $(FF_PYTHON_USE_INDEX_LOADER)), 1)
CC_FLAGS += -DFF_PYTHON_USE_INDEX_LOADER
NVCC_FLAGS += -DFF_PYTHON_USE_INDEX_LOADER
endif

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -Wl,-force_load,liblegion.a
else
	LD_FLAGS += -Wl,--whole-archive -llegion -Wl,--no-whole-archive
endif

NO_BUILD_ALL=1
.PHONY: all
all: $(OUTFILE) flexflow/core/legion_cffi.py 

include $(FF_HOME)/FlexFlow.mk

DEFINE_HEADERS_DIR ?= ./

flexflow/core/legion_cffi.py: legion_cffi.py.in legion_cffi_build.py
	$(PYTHON_EXE) legion_cffi_build.py --runtime-dir $(LG_RT_DIR) --defines-dir $(DEFINE_HEADERS_DIR) --output-dir flexflow/core

clean::
	$(RM) -f $(OUTFILE) $(SLIB_LEGION) $(SLIB_REALM) $(GEN_OBJS) $(GEN_GPU_OBJS) $(REALM_OBJS) $(LEGION_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) legion_defines.h realm_defines.h flexflow/core/legion_cffi.py *.pyc
	$(RM) -rf build dist *.egg-info
