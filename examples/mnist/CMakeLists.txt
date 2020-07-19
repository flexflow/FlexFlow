cmake_minimum_required(VERSION 3.1)

project(FlexFlowExample_Mnist)
set(project_target mnist)

set(CPU_SOURCES
  mnist.cc)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

cuda_add_executable(${project_target} ${CPU_SOURCES})
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} flexflow)

