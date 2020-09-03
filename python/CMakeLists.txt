cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Python)
set(project_target flexflow_python)

set(CPU_SRC
  flexflow_c.cc
  flexflow_dataloader.cc
  main.cc)

set(GPU_SRC
  flexflow_dataloader.cu)

cuda_add_executable(${project_target} ${CPU_SRC} ${GPU_SRC})
set_target_properties(${project_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow legion realm -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
