cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

#enable_language(CUDA)

include(ExternalProject) 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(FLEXFLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(LEGION_ROOT ${CMAKE_CURRENT_LIST_DIR}/deps/legion)

# Set a default build type if none was specified
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

# build shared or static flexflow lib
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" ON)

# option for using Python
option(FF_USE_PYTHON "Enable Python" ON)  

# option for using Python
option(FF_USE_GASNET "Run FlexFlow with GASNet" OFF)  

# option for cuda arch
set(CUDA_ARCH "" CACHE STRING "Target CUDA Arch")

set(FLEXFLOW_EXT_LIBRARIES "")

# ZLIB
include(zlib)

set(MAX_DIM 4)

# CUDA
include(cuda)
list(JOIN CUDA_GENCODE " " CUDA_GENCODE_FLAG)
set(CMAKE_CUDA_FLAGS ${CUDA_GENCODE_FLAG})

#CUDNN
include(cudnn)

# GASNet
# if(FF_USE_GASNET) 
#   include(gasnet)
# endif()

# Legion
include(legion)

# Start build FlexFlow

# get FLAGS from ENV
set(CC_FLAGS $ENV{CC_FLAGS})
set(NVCC_FLAGS $ENV{NVCC_FLAGS})
set(LD_FLAGS $ENV{LD_FLAGS})

list(APPEND CC_FLAGS
  -std=c++11
  -DMAX_TENSOR_DIM=${MAX_DIM})
  
list(APPEND NVCC_FLAGS
  -Wno-deprecated-gpu-targets
  -std=c++11
  -DMAX_TENSOR_DIM=${MAX_DIM})
  
list(APPEND LD_FLAGS
  -lrt
  -ldl
  -rdynamic)

# Set FLAGS
add_compile_options(${CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})
link_libraries(${LD_FLAGS})

# Python
if(FF_USE_PYTHON)
  add_compile_options(-DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON)
  add_subdirectory(python)
endif()

set(FLEXFLOW_INCLUDE_DIRS
  ${FLEXFLOW_ROOT}/include
  ${FLEXFLOW_ROOT}/
  ${CUDA_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIRS})

set(FLEXFLOW_HDR
  ${FLEXFLOW_ROOT}/include/accessor.h
  ${FLEXFLOW_ROOT}/include/config.h
  ${FLEXFLOW_ROOT}/include/cuda_helper.h
  ${FLEXFLOW_ROOT}/include/ffconst.h
  ${FLEXFLOW_ROOT}/include/initializer.h
  ${FLEXFLOW_ROOT}/include/loss_functions.h
  ${FLEXFLOW_ROOT}/include/mapper.h
  ${FLEXFLOW_ROOT}/include/metrics_functions.h
  ${FLEXFLOW_ROOT}/include/model.h
  ${FLEXFLOW_ROOT}/include/optimizer.h
  ${FLEXFLOW_ROOT}/include/simulator.h)  

set(FLEXFLOW_SRC
  ${FLEXFLOW_ROOT}/src/mapper/mapper.cc
  ${FLEXFLOW_ROOT}/src/ops/embedding.cc
  ${FLEXFLOW_ROOT}/src/metrics_functions/metrics_functions.cc
  ${FLEXFLOW_ROOT}/src/runtime/initializer.cc
  ${FLEXFLOW_ROOT}/src/runtime/model.cc
  ${FLEXFLOW_ROOT}/src/runtime/optimizer.cc
  ${FLEXFLOW_ROOT}/src/runtime/strategy.cc
  ${FLEXFLOW_ROOT}/src/runtime/simulator.cc)

set(FLEXFLOW_GPU_SRC
  ${FLEXFLOW_ROOT}/src/loss_functions/loss_functions.cu
  ${FLEXFLOW_ROOT}/src/metrics_functions/metrics_functions.cu
  ${FLEXFLOW_ROOT}/src/ops/attention.cu
  ${FLEXFLOW_ROOT}/src/ops/batch_matmul.cu
  ${FLEXFLOW_ROOT}/src/ops/batch_norm.cu
  ${FLEXFLOW_ROOT}/src/ops/concat.cu
  ${FLEXFLOW_ROOT}/src/ops/conv_2d.cu
  ${FLEXFLOW_ROOT}/src/ops/dropout.cu
  ${FLEXFLOW_ROOT}/src/ops/element_binary.cu
  ${FLEXFLOW_ROOT}/src/ops/element_unary.cu
  ${FLEXFLOW_ROOT}/src/ops/embedding.cu
  ${FLEXFLOW_ROOT}/src/ops/flat.cu
  ${FLEXFLOW_ROOT}/src/ops/fused.cu
  ${FLEXFLOW_ROOT}/src/ops/linear.cu
  ${FLEXFLOW_ROOT}/src/ops/pool_2d.cu
  ${FLEXFLOW_ROOT}/src/ops/reshape.cu
  ${FLEXFLOW_ROOT}/src/ops/reverse.cu
  ${FLEXFLOW_ROOT}/src/ops/softmax.cu
  ${FLEXFLOW_ROOT}/src/ops/split.cu
  ${FLEXFLOW_ROOT}/src/ops/topk.cu
  ${FLEXFLOW_ROOT}/src/ops/transpose.cu
  ${FLEXFLOW_ROOT}/src/runtime/accessor_kernel.cu
  ${FLEXFLOW_ROOT}/src/runtime/cuda_helper.cu
  ${FLEXFLOW_ROOT}/src/runtime/initializer_kernel.cu
  ${FLEXFLOW_ROOT}/src/runtime/model.cu
  ${FLEXFLOW_ROOT}/src/runtime/optimizer_kernel.cu
  ${FLEXFLOW_ROOT}/src/runtime/simulator.cu)

set(FLEXFLOW_CPP_DRV_SRC
  ${FLEXFLOW_ROOT}/src/runtime/cpp_driver.cc)  

# compile flexflow lib
if(BUILD_SHARED_LIBS)
   cuda_add_library(flexflow SHARED ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
 else()
   cuda_add_library(flexflow STATIC ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
 endif()

target_link_libraries(flexflow Legion)

target_include_directories(flexflow PUBLIC ${FLEXFLOW_INCLUDE_DIRS})

option(FF_BUILD_RESNET "build resnet example" OFF)
option(FF_BUILD_ALEXNET "build alexnet example" OFF)
option(FF_BUILD_DLRM "build DLRM example" OFF)
option(FF_BUILD_INCEPTION "build inception example" OFF)
option(FF_BUILD_CANDLE_UNO "build candle uno example" OFF)
option(FF_BUILD_TRANSFORMER "build transformer example" ON)
option(FF_BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)

if(FF_BUILD_RESNET OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/ResNet)
endif()

if(FF_BUILD_ALEXNET OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/AlexNet)
endif()

if(FF_BUILD_INCEPTION OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/InceptionV3)
endif()

#TODO: Once functional add to BUILD_ALL_EXAMPLES
if(FF_BUILD_CANDLE_UNO)
  add_subdirectory(examples/cpp/candle_uno)
endif()

if(FF_BUILD_DLRM OR BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/DLRM)

  add_executable(generate_dlrm_hetero_strategy src/runtime/dlrm_strategy_hetero.cc)
  target_include_directories(generate_dlrm_hetero_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})

  add_executable(generate_dlrm_strategy src/runtime/dlrm_strategy.cc)
  target_include_directories(generate_dlrm_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
endif()

if(FF_BUILD_TRANSFORMER OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/Transformer)
endif()

# installation
set(INCLUDE_DEST "include")
set(LIB_DEST "lib")
install(FILES ${FLEXFLOW_HDR} DESTINATION ${INCLUDE_DEST})
install(TARGETS flexflow DESTINATION ${LIB_DEST})
