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
project (legion-backend)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)
# FIXME what options should be set?
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
endif() # TRITON_ENABLE_GPU

#
# Protobuf
#
set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

#
# Legion backend
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libtriton_legion.ldscript libtriton_legion.ldscript COPYONLY)
find_package(Legion REQUIRED)

# Use customized protoc command to generate cpp files with proper layout
set(PROTO_SRCS onnx/onnx-data.pb.cc onnx/onnx-ml.pb.cc onnx/onnx-operators-ml.pb.cc)
add_custom_command(
  OUTPUT ${PROTO_SRCS}
  ALL
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    -I${CMAKE_CURRENT_SOURCE_DIR} --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    onnx/*.proto
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Compiling cpp files of the ONNX protos"
)

file(GLOB OPERATOR_SRCS operators/*.cc)
if(${TRITON_ENABLE_GPU})
  file(GLOB OPERATOR_CUDA_SRCS operators/*.cu)
  set(OPERATOR_SRCS ${OPERATOR_SRCS} ${OPERATOR_CUDA_SRCS})
endif() # TRITON_ENABLE_GPU


add_library(
  triton-legion-backend SHARED
  backend.cc
  model.cc
  instance.cc
  onnx_parser.cc
  ${PROTO_SRCS}
  runtime.cc
  operator.cc
  strategy.cc
  tensor.cc
  ${OPERATOR_SRCS}
)

add_library(
  TritonLegionBackend::triton-legion-backend ALIAS triton-legion-backend
)

target_include_directories(
  triton-legion-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(triton-legion-backend PRIVATE cxx_std_11)
# Note that __CUDA_NO_HALF_OPERATORS__ is for Legion's fp16 implementation,
# it was defined in legion's code but it was ignored in this CMake build
target_compile_options(
  triton-legion-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
  $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDA_NO_HALF_OPERATORS__>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-legion-backend
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE LEGION_USE_CUDA=1
  )
  # Some cuda_fp16 functions are only defined for __CUDA_ARCH__ >= 530,
  # default is 520 which is too old
  # FIXME expose target arch as CMake option
  set_target_properties(triton-legion-backend PROPERTIES CUDA_ARCHITECTURES "70")
endif() # TRITON_ENABLE_GPU

set_target_properties(
  triton-legion-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_legion
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_legion.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_legion.ldscript"
)

# Must enforce specific linking order so that the backend calls to Legion's
# hijacked cudart APIs and then the regular APIs if not hijacked.
target_link_libraries(
  triton-legion-backend
  PRIVATE
    Legion::Legion
)
if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-legion-backend
    PRIVATE
      CUDA::cudart
      -lcublas
      -lcudnn
  )
endif() # TRITON_ENABLE_GPU
target_link_libraries(
  triton-legion-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    protobuf::libprotobuf
)


# add_dependencies(triton-legion-backend legion_proto)

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOnnxRuntimeBackend)

install(
  TARGETS
    triton-legion-backend
  EXPORT
    triton-legion-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/legion
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/legion
)

if(${TRITON_LEGION_BACKEND_BUILD_TEST})
  add_subdirectory(test test)
endif()
