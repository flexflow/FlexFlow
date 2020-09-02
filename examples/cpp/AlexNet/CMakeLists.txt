cmake_minimum_required(VERSION 3.1)

project(FlexFlowExample_AlexNet)
set(project_target alexnet)

set(CPU_SOURCES
  alexnet.cc
  alexnet.cu
  alexnet.h)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets --cudart=static)

cuda_add_executable(${project_target} ${CPU_SOURCES})
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} flexflow)
