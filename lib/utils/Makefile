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
# Compiler and linker settings

CXX = g++

# TODO remove -Wno-unused-parameter -Wno-infinite-recursion -Wno-delete-abstract-non-virtual-dtor -Wno-sign-compare -Wno-reorder-ctor -Wno-logical-op-parentheses -Wno-delete-non-abstract-non-virtual-dtor once warnings fixed 
CXXFLAGS = -Wall -Wextra -Werror -pedantic -O2 -std=c++11 -Wno-unused-parameter -Wno-infinite-recursion -Wno-delete-abstract-non-virtual-dtor -Wno-sign-compare -Wno-reorder-ctor -Wno-logical-op-parentheses -Wno-delete-non-abstract-non-virtual-dtor

# Library name and source files
LIB_NAME = utils
SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/**/*.cc)
HDRS = $(wildcard $(INCLUDE_DIR)/**/*.h)

# Object files
OBJS = $(SRCS:.cc=.o)

# Include directories
INCLUDE_DIRS = -Iinclude -I../../deps/optional/include -I../../deps/visit_struct/include -I../../deps/variant/include

# Build rules
all: $(LIB_NAME).so

$(LIB_NAME).so: $(OBJS)
	$(CXX) -shared $(CXXFLAGS) $^ -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cc $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB_NAME).so

.PHONY: all clean install
