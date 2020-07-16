cmake_minimum_required(VERSION 3.1)

project(FlexFlowExample_Inception)
set(project_target inception)

set(CPU_SOURCES
  inception.cc
  inception.h)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

cuda_add_executable(${project_target} ${CPU_SOURCES})
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} flexflow)

