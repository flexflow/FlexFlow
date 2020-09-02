cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Python)
set(project_target flexflow_python)

set(CPU_SRC
  flexflow_c.cc
  flexflow_dataloader.cc
  main.cc)

set(GPU_SRC
  flexflow_dataloader.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -std=c++11)

cuda_add_executable(${project_target} ${CPU_SRC} ${GPU_SRC})
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow legion realm -Wl,--no-whole-archive ${Protobuf_LIBRARIES} ${FLEXFLOW_EXT_LIBRARIES}  "-lz -lrt -ldl -rdynamic")
