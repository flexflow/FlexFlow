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
USE_HIP         ?= 0
HIP_TARGET      ?= CUDA
USE_GPU_REDUCTIONS ?= 0
FF_USE_NCCL  ?= 0

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
ifeq ($(strip $(HIP_TARGET)),CUDA)
GEN_HIP_SRC	  ?= flexflow_dataloader.cu # .cu files
endif

PYBIND_INC    = $(shell $$PYTHON_EXE -m pybind11 --includes)
PYBIND_EXT    = $(shell $$PYTHON_EXE-config --extension-suffix)
FF_BYBIND_LIB = flexflow/core/flexflow_pybind11_internal$(PYBIND_EXT)
INC_FLAGS     += -I${FF_HOME}
CC_FLAGS      += -fPIC
NVCC_FLAGS    += -Xcompiler -fPIC
HIPCC_FLAGS    += -Xcompiler -fPIC

CC_FLAGS	  ?= -DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON 

#INC_FLAGS       += -DDISABLE_LEGION_CUDA_HIJACK 
  
ifeq ($(strip $(FF_ENABLE_DEBUG)), 1)
CC_FLAGS   += -DFF_DEBUG
NVCC_FLAGS += -DFF_DEBUG
HIPCC_FLAGS += -DFF_DEBUG
endif

ifeq ($(strip $(FF_PYTHON_USE_INDEX_LOADER)), 1)
CC_FLAGS   += -DFF_PYTHON_USE_INDEX_LOADER
NVCC_FLAGS += -DFF_PYTHON_USE_INDEX_LOADER
HIPCC_FLAGS += -DFF_PYTHON_USE_INDEX_LOADER
endif

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -Wl,-force_load,liblegion.a
else
	LD_FLAGS += -Wl,--whole-archive -llegion -Wl,--no-whole-archive
endif

FF_USE_PYTHON = $(USE_PYTHON)

include $(FF_HOME)/FlexFlow.mk

NO_BUILD_ALL=1
.PHONY: all
all: $(OUTFILE) flexflow/core/legion_cffi.py $(FF_BYBIND_LIB)

DEFINE_HEADERS_DIR ?= ./

flexflow/core/legion_cffi.py: legion_cffi.py.in legion_cffi_build.py
	$(PYTHON_EXE) legion_cffi_build.py --runtime-dir $(LG_RT_DIR) --defines-dir $(DEFINE_HEADERS_DIR) --output-dir flexflow/core

ifeq ($(strip $(FLEXFLOW_PYTHON)), 1)
$(FF_BYBIND_LIB): bindings.cc
	$(CC) $(CC_FLAGS) $(INC_FLAGS) $(PYBIND_INC) -shared -lstdc++ -std=c++11 -fPIC $< -o $@
else
$(FF_BYBIND_LIB): bindings.cc $(APP_OBJS) $(SLIB_LEGION) $(SLIB_REALM)
	$(CC) $(CC_FLAGS) $(INC_FLAGS) $(PYBIND_INC) -shared -lstdc++ -L$(FF_HOME)/python $(LD_FLAGS) $(LEGION_LD_FLAGS) bindings.cc $(APP_OBJS) $(SLIB_LEGION) $(SLIB_REALM) -o $@
endif

clean::
	$(RM) -f $(FF_BYBIND_LIB) legion_defines.h realm_defines.h flexflow/core/legion_cffi.py *.pyc
	$(RM) -rf build dist *.egg-info
