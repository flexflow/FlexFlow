cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

# Briefly change lib suffixes to get static zlib
set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
SET(BUILD_SHARED_LIBS OFF)
find_package(ZLIB)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})

option(FLOW_USE_EXTERNAL_PROTOBUF "use external protobuf" OFF)
if(NOT ${FLOW_USE_EXTERNAL_PROTOBUF})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
  include(protobuf)
else()
  add_library(protobuf STATIC IMPORTED GLOBAL)
endif()
option(FLOW_USE_EXTERNAL_LEGION "use external legion" OFF)
if (NOT ${FLOW_USE_EXTERNAL_LEGION})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
  include(legion)
endif()

include(FindProtobuf)
include(FindCUDA)
set(FLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})
protobuf_generate_cpp(FLOW_PB_SRC FLOW_PB_HDR ${FLOW_ROOT}/src/runtime/strategy.proto)
get_filename_component(FLOW_PB_INC_DIR ${FLOW_PB_HDR} DIRECTORY)

set(FLOW_INCLUDE
  ${FLOW_ROOT}/include
  ${FLOW_PB_INC_DIR}
  ${FLOW_ROOT}/
  /usr/local/cuda/include/)

set(FLOW_SRC
  ${FLOW_PB_SRC}
  ${FLOW_ROOT}/src/mapper/mapper.cc
  ${FLOW_ROOT}/src/ops/embedding.cc
  ${FLOW_ROOT}/src/runtime/initializer.cc
  ${FLOW_ROOT}/src/runtime/model.cc
  ${FLOW_ROOT}/src/runtime/optimizer.cc
  ${FLOW_ROOT}/src/runtime/strategy.cc)

set(FLOW_GPU_SRC
  ${FLOW_ROOT}/src/ops/batch_norm.cu
  ${FLOW_ROOT}/src/ops/concat.cu
  ${FLOW_ROOT}/src/ops/conv_2d.cu
  ${FLOW_ROOT}/src/ops/element_binary.cu
  ${FLOW_ROOT}/src/ops/element_unary.cu
  ${FLOW_ROOT}/src/ops/embedding.cu
  ${FLOW_ROOT}/src/ops/flat.cu
  ${FLOW_ROOT}/src/ops/linear.cu
  ${FLOW_ROOT}/src/ops/mse_loss.cu
  ${FLOW_ROOT}/src/ops/pool_2d.cu
  ${FLOW_ROOT}/src/ops/softmax.cu
  ${FLOW_ROOT}/src/runtime/accessor_kernel.cu
  ${FLOW_ROOT}/src/runtime/cuda_helper.cu
  ${FLOW_ROOT}/src/runtime/initializer_kernel.cu
  ${FLOW_ROOT}/src/runtime/model.cu
  ${FLOW_ROOT}/src/runtime/optimizer_kernel.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
cuda_add_library(flexflow ${FLOW_GPU_SRC} ${FLOW_SRC})
target_include_directories(flexflow PUBLIC ${FLOW_INCLUDE})

set(protobuf_lib_name protobuf)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(protobuf_lib_name protobufd)
endif()
target_link_libraries(flexflow Legion::Legion ${protobuf_lib_name} cudnn cublas curand  "-L /usr/local/cuda/lib64")

option(BUILD_RESNET "build resnet example" OFF)
option(BUILD_ALEXNET "build alexnet example" OFF)
option(BUILD_ALEXNET_NEWAPI "build alexnet_newapi example" OFF)
option(BUILD_DLRM "build DLRM example" OFF)
option(BUILD_INCEPTION "build inception example" OFF)
option(BUILD_CANDLE_UNO "build candle uno example" OFF)
option(BUILD_PCA "build PCA example" OFF)
option(BUILD_ALL_EXAMPLES "build all examples. Overrides others" ON)

if(${BUILD_RESNET} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/ResNet)
endif()

if(${BUILD_ALEXNET} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/AlexNet)
endif()

if(${BUILD_ALEXNET_NEWAPI} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/AlexNet_newapi)
endif()

if(${BUILD_INCEPTION} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/InceptionV3)
endif()

if(${BUILD_PCA} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/PCA)
endif()

#TODO: Once functional add to BUILD_ALL_EXAMPLES
if(${BUILD_CANDLE_UNO})
  add_subdirectory(examples/candle_uno)
endif()

if(${BUILD_DLRM} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/DLRM)

  add_executable(generate_dlrm_hetero_strategy src/runtime/dlrm_strategy_hetero.cc ${FLOW_PB_SRC})
  target_include_directories(generate_dlrm_hetero_strategy PUBLIC ${FLOW_INCLUDE})
  target_link_libraries(generate_dlrm_hetero_strategy ${protobuf_lib_name})
  
  add_executable(generate_dlrm_strategy src/runtime/dlrm_strategy.cc ${FLOW_PB_SRC})
  target_include_directories(generate_dlrm_strategy PUBLIC ${FLOW_INCLUDE})
  target_link_libraries(generate_dlrm_strategy ${protobuf_lib_name})
endif()

if(${BUILD_MNIST} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/mnist)
endif()
