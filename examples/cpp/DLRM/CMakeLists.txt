  
cmake_minimum_required(VERSION 3.1)

project(FlexFlowExample_ResNet)
set(project_target dlrm)
find_package(HDF5 REQUIRED COMPONENTS C)
set(CPU_SOURCES
  dlrm.cc
  dlrm.cu
  dlrm.h)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

cuda_add_executable(${project_target} ${CPU_SOURCES})
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR} ${HDF5_C_INCLUDE_DIRS})
target_link_libraries(${project_target} flexflow ${HDF5_C_LIBRARIES})
