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

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 5		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)
USE_HIP         ?= 0		# Include HIP support (requires HIP)
HIP_TARGET      ?= ROCM
USE_GPU_REDUCTIONS ?= 0
FF_USE_NCCL     ?= 0
USE_PYTHON      := 1
FF_USE_PYTHON   := 1
SHARED_OBJECTS  := 1     # we build the shared lib for legion
# FF_PYTHON_USE_INDEX_LOADER = 1

INSTALL_TOKENIZERS := $(shell $(FF_HOME)/scripts/install_tokenizer.sh)

ifeq ($(shell uname -s), Darwin)
  PYTHON_EXT := dylib
else
  PYTHON_EXT := so
endif

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

$(info    VAR is $(PYTHON_EXE))

FF_HOME ?= ../
LG_RT_DIR ?= $(FF_HOME)/deps/legion/runtime
LG_PYTHON_BINDINGS_DIR ?= $(FF_HOME)/deps/legion/bindings/python

# Put the binary file name here
LEGION_PYTHON_BIN		  ?= legion_python

GEN_SRC		  ?= $(LG_PYTHON_BINDINGS_DIR)/main.cc # .cc files

CC_FLAGS	  ?= -DBINDINGS_AUGMENT_PYTHONPATH

FF_LIB = libflexflow.$(PYTHON_EXT)
LEGION_CANONICAL_PYTHON_LIB := liblegion_canonical_python.$(PYTHON_EXT)

INC_FLAGS     ?= -I${FF_HOME}
LD_FLAGS      ?=
CC_FLAGS      ?= -fPIC -DBINDINGS_AUGMENT_PYTHONPATH
NVCC_FLAGS    ?= -Xcompiler -fPIC
ifeq ($(strip $(USE_HIP)),1)
ifeq ($(strip $(HIP_TARGET)),CUDA)
HIPCC_FLAGS    ?= -Xcompiler -fPIC
else
HIPCC_FLAGS    ?= -fPIC
endif
endif

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

include $(FF_HOME)/FlexFlow.mk

NO_BUILD_ALL=1
.PHONY: all
all: $(LEGION_PYTHON_BIN) flexflow/core/flexflow_cffi_header.py $(FF_LIB) legion_info.py legion_builtin_cffi.py legion_canonical_cffi.py $(LEGION_CANONICAL_PYTHON_LIB) legion_top.py legion_cffi.py

$(LEGION_PYTHON_BIN) : $(LG_PYTHON_BINDINGS_DIR)/main.cc.o $(SLIB_LEGION) $(SLIB_REALM)
	@echo "---> Linking objects into one binary: $(LEGION_PYTHON_BIN)"
	$(CXX) -o $(LEGION_PYTHON_BIN) $(LG_PYTHON_BINDINGS_DIR)/main.cc.o $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS)

legion_builtin_cffi.py: $(LG_PYTHON_BINDINGS_DIR)/legion_cffi.py.in $(LG_PYTHON_BINDINGS_DIR)/legion_cffi_build.py $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(PYTHON_EXE) $(LG_PYTHON_BINDINGS_DIR)/legion_cffi_build.py --defines-dir $(DEFINE_HEADERS_DIR) --output-dir .

legion_canonical_cffi.py: $(LG_PYTHON_BINDINGS_DIR)/legion_cffi.py.in $(LG_PYTHON_BINDINGS_DIR)/legion_cffi_build.py $(LEGION_DEFINES_HEADER) $(REALM_DEFINES_HEADER)
	$(PYTHON_EXE) $(LG_PYTHON_BINDINGS_DIR)/legion_cffi_build.py --canonical --defines-dir $(DEFINE_HEADERS_DIR) --libname $(LEGION_CANONICAL_PYTHON_LIB) --output-dir .

legion_info.py: $(LG_PYTHON_BINDINGS_DIR)/legion_info_build.py
	$(PYTHON_EXE) $(LG_PYTHON_BINDINGS_DIR)/legion_info_build.py

# build the shared lib for canonical python
$(LEGION_CANONICAL_PYTHON_LIB): $(LG_PYTHON_BINDINGS_DIR)/canonical_python.cc $(SLIB_LEGION) $(SLIB_REALM)
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -shared $< $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) -o $@

# the following two are copy
legion_top.py: $(LG_PYTHON_BINDINGS_DIR)/legion_top.py
	cp -f $< $@

legion_cffi.py: $(LG_PYTHON_BINDINGS_DIR)/legion_cffi.py
	cp -f $< $@
	
flexflow/core/flexflow_cffi_header.py: flexflow_cffi_header.py.in flexflow_cffi_build.py $(FF_HOME)/include/flexflow/flexflow_c.h
	$(PYTHON_EXE) flexflow_cffi_build.py --ffhome-dir $(FF_HOME) --output-dir flexflow/core

# build the .so for flexflow
# hack: we do not need main.cc anymore
FF_SHARED_OBJS := $(filter-out $(LG_PYTHON_BINDINGS_DIR)/main.cc.o,$(APP_OBJS))
$(FF_LIB): $(FF_SHARED_OBJS) $(SLIB_LEGION) $(SLIB_REALM)
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -shared -lstdc++ -L$(FF_HOME)/python $(FF_SHARED_OBJS) $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) -o $@

clean::
	$(RM) -f -f $(FF_LIB) $(LEGION_PYTHON_BIN) $(SLIB_LEGION) $(SLIB_REALM) $(APP_OBJS) $(REALM_OBJS) $(LEGION_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS) $(ASM_OBJS) $(LEGION_CANONICAL_PYTHON_LIB) legion_defines.h realm_defines.h legion_builtin_cffi.py legion_canonical_cffi.py legion_info.py legion_top.py legion_cffi.py flexflow/core/flexflow_cffi_header.py *.pyc
	$(RM) -rf build dist *.egg-info
