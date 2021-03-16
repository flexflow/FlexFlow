cmake_minimum_required(VERSION 3.10)

project(FlexFlowTests)
set(project_target unit-test)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES False *.cc)

set(CPU_SRC
  ${TEST_SOURCES}
  ${FLEXFLOW_SRC}
  )

set(GPU_SRC
  ${FLEXFLOW_GPU_SRC}
  )

cuda_add_executable(${project_target} ${CPU_SRC} ${GPU_SRC})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR} ${gtest_SOURCE_DIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES} gtest gtest_main)
add_test(UnitTests ./${project_target})
