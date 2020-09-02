cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

include(ExternalProject) 

# Briefly change lib suffixes to get static zlib
# set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
# SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

list(APPEND CC_FLAGS
  -std=c++11)
  
list(APPEND NVCC_FLAGS
  -Wno-deprecated-gpu-targets
  -std=c++11)

# find_package(ZLIB)
# set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})

# CUDA and CUDNN
include(cudacudnn)

# Google Protobuf
option(FlexFlow_USE_External_Protobuf "Use external Protobuf" ON)
include(protobuf)

# Legion
option(FlexFlow_USE_External_Legion "Use external Legion" ON)
include(legion)

# set(CUDA_BIN_PATH /spack/opt/spack/linux-scientific7-x86_64/gcc-7.3.0/cuda-10.0.130-etoqn3vlwx2os25n3rmtksntnu5ioorb)

set(FLEXFLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})
protobuf_generate_cpp(FLEXFLOW_PB_SRC FLOW_PB_HDR ${FLEXFLOW_ROOT}/src/runtime/strategy.proto)
get_filename_component(FLOW_PB_INC_DIR ${FLOW_PB_HDR} DIRECTORY)

set(FLEXFLOW_INCLUDE_DIRS
  ${FLEXFLOW_ROOT}/include
  ${FLOW_PB_INC_DIR}
  ${FLEXFLOW_ROOT}/
  ${CUDA_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIRS})

set(FLEXFLOW_SRC
  ${FLEXFLOW_PB_SRC}
  ${FLEXFLOW_ROOT}/src/mapper/mapper.cc
  ${FLEXFLOW_ROOT}/src/ops/embedding.cc
  ${FLEXFLOW_ROOT}/src/metrics_functions/metrics_functions.cc
  ${FLEXFLOW_ROOT}/src/runtime/initializer.cc
  ${FLEXFLOW_ROOT}/src/runtime/model.cc
  ${FLEXFLOW_ROOT}/src/runtime/optimizer.cc
  ${FLEXFLOW_ROOT}/src/runtime/strategy.cc
  ${FLEXFLOW_ROOT}/src/runtime/simulator.cc)

set(FLEXFLOW_GPU_SRC
  ${FLEXFLOW_ROOT}/src/ops/batch_norm.cu
  ${FLEXFLOW_ROOT}/src/ops/concat.cu
  ${FLEXFLOW_ROOT}/src/ops/conv_2d.cu
  ${FLEXFLOW_ROOT}/src/ops/element_binary.cu
  ${FLEXFLOW_ROOT}/src/ops/element_unary.cu
  ${FLEXFLOW_ROOT}/src/ops/embedding.cu
  ${FLEXFLOW_ROOT}/src/ops/flat.cu
  ${FLEXFLOW_ROOT}/src/ops/linear.cu
  ${FLEXFLOW_ROOT}/src/ops/pool_2d.cu
  ${FLEXFLOW_ROOT}/src/ops/softmax.cu
  ${FLEXFLOW_ROOT}/src/ops/dropout.cu
  ${FLEXFLOW_ROOT}/src/metrics_functions/metrics_functions.cu
  ${FLEXFLOW_ROOT}/src/loss_functions/loss_functions.cu
  ${FLEXFLOW_ROOT}/src/runtime/accessor_kernel.cu
  ${FLEXFLOW_ROOT}/src/runtime/cuda_helper.cu
  ${FLEXFLOW_ROOT}/src/runtime/initializer_kernel.cu
  ${FLEXFLOW_ROOT}/src/runtime/model.cu
  ${FLEXFLOW_ROOT}/src/runtime/optimizer_kernel.cu
  ${FLEXFLOW_ROOT}/src/runtime/simulator.cu)

option(FlexFlow_USE_Python "Use Python" ON)  
if(FlexFlow_USE_Python)
  add_compile_options(-DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON)
  add_subdirectory(python)
 # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -std=c++11)
 #
 #  list(APPEND FLEXFLOW_INCLUDE_DIRS
 #    ${FLEXFLOW_ROOT}/python)
 #
 #  list(APPEND FLEXFLOW_SRC
 #    ${FLEXFLOW_ROOT}/python/flexflow_c.cc
 #    ${FLEXFLOW_ROOT}/python/flexflow_dataloader.cc
 #    ${FLEXFLOW_ROOT}/python/main.cc)
 #
 #  list(APPEND FLEXFLOW_GPU_SRC
 #    ${FLEXFLOW_ROOT}/python/flexflow_dataloader.cu)
 #
 #  cuda_add_executable(flexflow_python ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC})
 #  target_include_directories(flexflow_python PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
 #  add_dependencies(flexflow_python ${LEGION_NAME})
 #  target_link_libraries(flexflow_python -Wl,--whole-archive legion realm -Wl,--no-whole-archive ${Protobuf_LIBRARIES} ${FLEXFLOW_EXT_LIBRARIES}  "-lz -lrt -ldl -rdynamic")
endif()

add_compile_options(${CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})
cuda_add_library(flexflow STATIC ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC})

target_include_directories(flexflow PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
add_dependencies(flexflow ${LEGION_NAME})

#target_link_libraries(flexflow)
#target_link_libraries(flexflow ${Protobuf_LIBRARIES} cudnn cublas curand  "-L/sw/cuda/10.1/lib64 -L/scratch/shared/wwu12/cudnn/lib64" ${INSTALL_DIR}/lib64/liblegion.a ${INSTALL_DIR}/lib64/librealm.a)

option(BUILD_RESNET "build resnet example" OFF)
option(BUILD_ALEXNET "build alexnet example" OFF)
option(BUILD_DLRM "build DLRM example" OFF)
option(BUILD_INCEPTION "build inception example" OFF)
option(BUILD_CANDLE_UNO "build candle uno example" OFF)
option(BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)

if(${BUILD_RESNET} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/cpp/ResNet)
endif()

if(${BUILD_ALEXNET} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/cpp/AlexNet)
endif()

if(${BUILD_INCEPTION} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/cpp/InceptionV3)
endif()

#TODO: Once functional add to BUILD_ALL_EXAMPLES
if(${BUILD_CANDLE_UNO})
  add_subdirectory(examples/cpp/candle_uno)
endif()

if(${BUILD_DLRM} OR ${BUILD_ALL_EXAMPLES})
  add_subdirectory(examples/cpp/DLRM)

  add_executable(generate_dlrm_hetero_strategy src/runtime/dlrm_strategy_hetero.cc ${FLEXFLOW_PB_SRC})
  target_include_directories(generate_dlrm_hetero_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
  target_link_libraries(generate_dlrm_hetero_strategy ${Protobuf_LIBRARIES})

  add_executable(generate_dlrm_strategy src/runtime/dlrm_strategy.cc ${FLEXFLOW_PB_SRC})
  target_include_directories(generate_dlrm_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
  target_link_libraries(generate_dlrm_strategy ${Protobuf_LIBRARIES})
endif()