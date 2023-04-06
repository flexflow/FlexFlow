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

cmake_minimum_required(VERSION 3.18)

project(tritonlegionbackend LANGUAGES C CXX)

include(GNUInstallDirs)

#
# Options
#
# To build the Legion backend you must :
#
#   - Point to the "legion/runtime" directory in a Legion repo using
#     TRITON_LEGION_LG_RT_DIR
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_LEGION_BACKEND_BUILD_TEST "Build the unit tests for the backend" ON)
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

set(GASNET_CONDUIT "mpi" CACHE STRING "Conduit used in Gasnet")
option(LEGION_USE_NETWORK "Include support for multi-node execution" ON)

if (WIN32)
  message(FATAL_ERROR
    "Legion backend is currently not supported for Windows")
endif() # WIN32

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Gasnet
#
include(ExternalProject)
if(${LEGION_USE_NETWORK})
  ExternalProject_Add(gasnet
    PREFIX gasnet
    GIT_REPOSITORY "https://github.com/StanfordLegion/gasnet.git"
    GIT_TAG "master"
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/gasnet/src/gasnet"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND $(MAKE) CONDUIT=${GASNET_CONDUIT}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    TEST_COMMAND ""
  )
set(GASNET_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gasnet/src/gasnet/release)
# Setting variables for Legion build
set(LEGION_NETWORKS gasnetex)
set(LEGION_DEPENDS gasnet)
endif()

#
# Legion
#
include(ExternalProject)
ExternalProject_Add(legion
  PREFIX legion
  GIT_REPOSITORY "https://gitlab.com/StanfordLegion/legion.git"
  GIT_TAG "control_replication"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/legion/src/legion"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_BUILD_TYPE:STRING=RELEASE
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DLegion_MAX_DIM:STRING=4
    -DLegion_MAX_FIELDS:STRING=64
    -DLegion_OUTPUT_LEVEL:STRING=INFO
    -DLegion_USE_CUDA:BOOL=${TRITON_ENABLE_GPU}
    -DLegion_USE_ZLIB:BOOL=ON
    -DLegion_USE_LIBD:BOOLL=ON
    -DBUILD_MARCH:STRING=native
    -DLegion_NETWORKS:STRING=${LEGION_NETWORKS}
    -DGASNet_CONDUIT:STRING=${GASNET_CONDUIT}
    -DGASNet_ROOT_DIR:PATH=${GASNET_ROOT}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/legion
    DEPENDS ${LEGION_DEPENDS}
)
# FIXME what CUDA arch to use?
# -DLegion_CUDA_ARCH=70
set(_FINDPACKAGE_LEGION_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/legion/share/Legion/cmake")

#
# Protobuf
#
ExternalProject_Add(protobuf-repo
  PREFIX protobuf-repo
  GIT_REPOSITORY "https://github.com/protocolbuffers/protobuf.git"
  GIT_TAG "v3.17.1"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf-repo/src/protobuf"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# Build protobuf project from protobuf-repo
#
# Use -DCMAKE_BUILD_TYPE:STRING=RELEASE to workaround the protobuf build issue
# described in https://stackoverflow.com/a/60218066/2091555
ExternalProject_Add(protobuf
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf-repo/src/protobuf/cmake"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -Dprotobuf_BUILD_TESTS:BOOL=OFF
    -Dprotobuf_WITH_ZLIB:BOOL=OFF
    -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -DCMAKE_BUILD_TYPE:STRING=RELEASE
    -DBUILD_SHARED_LIBS:STRING=no
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf
  DEPENDS protobuf-repo
)
set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/${CMAKE_INSTALL_LIBDIR}/cmake/protobuf")
set(TRITON_LEGION_PROTOBUF_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf)

#
# Build googletest project from protobuf-repo
#
if(${TRITON_LEGION_BACKEND_BUILD_TEST})
  ExternalProject_Add(googletest
    PREFIX protobuf
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf-repo/src/protobuf/third_party/googletest"
    DOWNLOAD_COMMAND ""
    CMAKE_CACHE_ARGS
      ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
      ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
      -DCMAKE_BUILD_TYPE:STRING=RELEASE
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/googletest
    DEPENDS protobuf-repo
  )
endif()

#
# Build Triton Legion Backend
#
set(BACKEND_DEPENDS protobuf legion)
if(${TRITON_LEGION_BACKEND_BUILD_TEST})
  set(BACKEND_DEPENDS ${BACKEND_DEPENDS} googletest)
endif()
ExternalProject_Add(triton-legion-backend
  PREFIX triton-legion-backend
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton-legion-backend"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -DLegion_DIR:PATH=${_FINDPACKAGE_LEGION_CONFIG_DIR}
    -DLEGION_ROOT:PATH=${CMAKE_CURRENT_BINARY_DIR}/legion
    -DGTEST_ROOT:PATH=${CMAKE_CURRENT_BINARY_DIR}/googletest
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
    -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
    -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
    -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG}
    -DTRITON_LEGION_BACKEND_BUILD_TEST:BOOL=${TRITON_LEGION_BACKEND_BUILD_TEST}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/triton-legion
  DEPENDS ${BACKEND_DEPENDS}
)

unset(CMAKE_INSTALL_PREFIX CACHE)
