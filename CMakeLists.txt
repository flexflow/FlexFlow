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

# find_package(ZLIB)
# set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})


include(cudacudnn)

# list(APPEND CMAKE_PREFIX_PATH /scratch/shared/wwu12/cudnn)
# find_package(CUDNN REQUIRED)


option(FLEXFLOW_USE_EXTERNAL_PROTOBUF "use external protobuf" ON)
include(protobuf)

option(FLEXFLOW_USE_EXTERNAL_LEGION "use external legion" OFF)
include(legion)

option(FlexFlow_USE_Python "Use Python" ON)

# set(CUDA_BIN_PATH /spack/opt/spack/linux-scientific7-x86_64/gcc-7.3.0/cuda-10.0.130-etoqn3vlwx2os25n3rmtksntnu5ioorb)

set(FLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})
protobuf_generate_cpp(FLOW_PB_SRC FLOW_PB_HDR ${FLOW_ROOT}/src/runtime/strategy.proto)
get_filename_component(FLOW_PB_INC_DIR ${FLOW_PB_HDR} DIRECTORY)

set(FLOW_INCLUDE
  ${FLOW_ROOT}/include
  ${FLOW_PB_INC_DIR}
  ${FLOW_ROOT}/
  ${CUDA_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIRS})

set(FLOW_SRC
  ${FLOW_PB_SRC}
  ${FLOW_ROOT}/src/mapper/mapper.cc
  ${FLOW_ROOT}/src/ops/embedding.cc
  ${FLOW_ROOT}/src/metrics_functions/metrics_functions.cc
  ${FLOW_ROOT}/src/runtime/initializer.cc
  ${FLOW_ROOT}/src/runtime/model.cc
  ${FLOW_ROOT}/src/runtime/optimizer.cc
  ${FLOW_ROOT}/src/runtime/strategy.cc
  ${FLOW_ROOT}/src/runtime/simulator.cc)

set(FLOW_GPU_SRC
  ${FLOW_ROOT}/src/ops/batch_norm.cu
  ${FLOW_ROOT}/src/ops/concat.cu
  ${FLOW_ROOT}/src/ops/conv_2d.cu
  ${FLOW_ROOT}/src/ops/element_binary.cu
  ${FLOW_ROOT}/src/ops/element_unary.cu
  ${FLOW_ROOT}/src/ops/embedding.cu
  ${FLOW_ROOT}/src/ops/flat.cu
  ${FLOW_ROOT}/src/ops/linear.cu
  ${FLOW_ROOT}/src/ops/pool_2d.cu
  ${FLOW_ROOT}/src/ops/softmax.cu
  ${FLOW_ROOT}/src/ops/dropout.cu
  ${FLOW_ROOT}/src/metrics_functions/metrics_functions.cu
  ${FLOW_ROOT}/src/loss_functions/loss_functions.cu
  ${FLOW_ROOT}/src/runtime/accessor_kernel.cu
  ${FLOW_ROOT}/src/runtime/cuda_helper.cu
  ${FLOW_ROOT}/src/runtime/initializer_kernel.cu
  ${FLOW_ROOT}/src/runtime/model.cu
  ${FLOW_ROOT}/src/runtime/optimizer_kernel.cu
  ${FLOW_ROOT}/src/runtime/simulator.cu)
  
if(FlexFlow_USE_Python)
  add_compile_options(-DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON)
  add_subdirectory(python)
 # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -std=c++11)
 #
 #  list(APPEND FLOW_INCLUDE
 #    ${FLOW_ROOT}/python)
 #
 #  list(APPEND FLOW_SRC
 #    ${FLOW_ROOT}/python/flexflow_c.cc
 #    ${FLOW_ROOT}/python/flexflow_dataloader.cc
 #    ${FLOW_ROOT}/python/main.cc)
 #
 #  list(APPEND FLOW_GPU_SRC
 #    ${FLOW_ROOT}/python/flexflow_dataloader.cu)
 #
 #  cuda_add_executable(flexflow_python ${FLOW_GPU_SRC} ${FLOW_SRC})
 #  target_include_directories(flexflow_python PUBLIC ${FLOW_INCLUDE})
 #  add_dependencies(flexflow_python ${LEGION_NAME})
 #  target_link_libraries(flexflow_python -Wl,--whole-archive legion realm -Wl,--no-whole-archive ${Protobuf_LIBRARIES} ${FLEXFLOW_EXT_LIBRARIES}  "-lz -lrt -ldl -rdynamic")
endif()

add_compile_options(-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -std=c++11)
cuda_add_library(flexflow STATIC ${FLOW_GPU_SRC} ${FLOW_SRC})

target_include_directories(flexflow PUBLIC ${FLOW_INCLUDE})
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

  add_executable(generate_dlrm_hetero_strategy src/runtime/dlrm_strategy_hetero.cc ${FLOW_PB_SRC})
  target_include_directories(generate_dlrm_hetero_strategy PUBLIC ${FLOW_INCLUDE})
  target_link_libraries(generate_dlrm_hetero_strategy ${Protobuf_LIBRARIES})

  add_executable(generate_dlrm_strategy src/runtime/dlrm_strategy.cc ${FLOW_PB_SRC})
  target_include_directories(generate_dlrm_strategy PUBLIC ${FLOW_INCLUDE})
  target_link_libraries(generate_dlrm_strategy ${Protobuf_LIBRARIES})
endif()