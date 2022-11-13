cmake_minimum_required(VERSION 3.10)

project(FlexFlowExampleInferenceGShard)
set(project_target g_shard)

set(CPU_SRC
  ${FLEXFLOW_CPP_DRV_SRC}
  g_shard.cc)

cuda_add_executable(${project_target} ${CPU_SRC})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
