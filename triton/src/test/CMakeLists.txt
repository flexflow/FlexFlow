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

cmake_minimum_required (VERSION 3.18)

#
# ONNX Parser
#
# Use customized protoc command to generate cpp files with proper layout
set(PROTO_SRCS onnx/onnx-data.pb.cc onnx/onnx-ml.pb.cc onnx/onnx-operators-ml.pb.cc)
add_custom_command(
  OUTPUT ${PROTO_SRCS}
  ALL
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    -I${CMAKE_CURRENT_SOURCE_DIR}/.. --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../onnx/*.proto
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Compiling cpp files of the ONNX protos"
)
set(
  ONNX_PARSER_SRCS
  ${PROTO_SRCS}
  ../onnx_parser.cc
  ../tensor.cc
  ../operator.cc
)

set(
  ONNX_PARSER_HDRS
  ../onnx_parser.h
)

file(GLOB MOCK_SRCS mock/* )
set(
  ONNX_PARSER_MOCK_SRCS
  ${MOCK_SRCS}
)

set(
  ONNX_PARSER_TEST_SRCS
  onnx_parser_test.cc
  ${ONNX_PARSER_SRCS}
  ${ONNX_PARSER_MOCK_SRCS}
)

set(
  ONNX_PARSER_TEST_HDRS
  ${ONNX_PARSER_HDRS}
)

find_package(GTest REQUIRED)
add_executable(
  onnx_parser_test
  ${ONNX_PARSER_TEST_SRCS}
  ${ONNX_PARSER_TEST_HDRS}
)
set_target_properties(
  onnx_parser_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  onnx_parser_test
  PRIVATE ${GTEST_INCLUDE_DIR}
  PRIVATE ${LEGION_ROOT}/include # Not using target as we only want declarations
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
  onnx_parser_test
  PRIVATE triton-core-serverapi      # from repo-core
  PRIVATE triton-backend-utils
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE protobuf::libprotobuf
  PRIVATE -L${CMAKE_CURRENT_SOURCE_DIR}/..
)
install(
  TARGETS onnx_parser_test
  RUNTIME DESTINATION test
)

# Test data
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
  DESTINATION test
)
