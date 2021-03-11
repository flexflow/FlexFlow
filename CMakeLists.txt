cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

include(ExternalProject) 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(FLEXFLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})

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
option(FF_USE_GASNET "Run FlexFlow with GASNet" ON)
set(FF_GASNET_CONDUITS udp mpi ibv)
set(FF_GASNET_CONDUIT "mpi" CACHE STRING "Select Gasnet conduit ${FF_GASNET_CONDUITS}")
set_property(CACHE FF_GASNET_CONDUIT PROPERTY STRINGS ${FF_GASNET_CONDUITS})  

# option for cuda arch
set(FF_CUDA_ARCH "" CACHE STRING "Target CUDA Arch")

# option for nccl
option(FF_USE_NCCL "Run FlexFlow with NCCL" ON)

set(FLEXFLOW_EXT_LIBRARIES "")
set(FLEXFLOW_INCLUDE_DIRS "")

# get FLAGS from ENV
set(CC_FLAGS $ENV{CC_FLAGS})
set(NVCC_FLAGS $ENV{NVCC_FLAGS})
set(LD_FLAGS $ENV{LD_FLAGS})

set(FF_MAX_DIM 4)

# ZLIB
include(zlib)

# CUDA
include(cuda)
 
# CUDNN
include(cudnn)

# NCCL
if(FF_USE_NCCL)
  include(nccl)
  list(APPEND CC_FLAGS
    -DFF_USE_NCCL)
  list(APPEND NVCC_FLAGS
    -DFF_USE_NCCL)
endif()

# Legion
include(legion)

if(FF_USE_PYTHON)
  list(APPEND CC_FLAGS
    -DBINDINGS_AUGMENT_PYTHONPATH)
  list(APPEND NVCC_FLAGS
    -DBINDINGS_AUGMENT_PYTHONPATH)
endif()

# Start build FlexFlow
if (CMAKE_BUILD_TYPE EQUAL "Debug")
  list(APPEND CC_FLAGS
    -DFF_DEBUG)
  list(APPEND NVCC_FLAGS
    -DFF_DEBUG)
endif()

list(APPEND CC_FLAGS
  -std=c++11
  -DMAX_TENSOR_DIM=${FF_MAX_DIM})
  
list(APPEND NVCC_FLAGS
  -Wno-deprecated-gpu-targets
  -std=c++11
  -DMAX_TENSOR_DIM=${FF_MAX_DIM})
  
list(APPEND LD_FLAGS
  -lrt
  -ldl
  -rdynamic)

# Set FLAGS
add_compile_options(${CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})
link_libraries(${LD_FLAGS})

list(APPEND FLEXFLOW_INCLUDE_DIRS
  ${FLEXFLOW_ROOT}/include
  ${FLEXFLOW_ROOT})

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
  ${FLEXFLOW_ROOT}/src/runtime/simulator.cc
  ${FLEXFLOW_ROOT}/src/runtime/machine_model.cc)

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

#message("FLEXFLOW_INCLUDE_DIRS: ${FLEXFLOW_INCLUDE_DIRS}")

# compile flexflow lib
if(BUILD_SHARED_LIBS)
   cuda_add_library(flexflow SHARED ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
 else()
   cuda_add_library(flexflow STATIC ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
 endif()

target_include_directories(flexflow PUBLIC ${FLEXFLOW_INCLUDE_DIRS})

if(FF_USE_NCCL)
   add_dependencies(flexflow ${NCCL_NAME})
 endif()

# build binary
option(FF_BUILD_RESNET "build resnet example" OFF)
option(FF_BUILD_ALEXNET "build alexnet example" OFF)
option(FF_BUILD_DLRM "build DLRM example" OFF)
option(FF_BUILD_INCEPTION "build inception example" OFF)
option(FF_BUILD_CANDLE_UNO "build candle uno example" OFF)
option(FF_BUILD_TRANSFORMER "build transformer example" OFF)
option(FF_BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)

# Python
if(FF_USE_PYTHON)
  if(FF_USE_PYBIND)
    add_subdirectory(deps/pybind11)
  endif()
  add_subdirectory(python)
endif()

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
