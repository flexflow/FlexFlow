cmake_minimum_required(VERSION 3.10)

project(FlexFlowExample_AlexNet)
set(project_target alexnet)

set(CPU_SOURCES
  alexnet.cc
  alexnet.cu
  alexnet.h)

cuda_add_executable(${project_target} ${CPU_SOURCES})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow legion realm -Wl,--no-whole-archive ${Protobuf_LIBRARIES} ${FLEXFLOW_EXT_LIBRARIES})
