cmake_minimum_required(VERSION 3.10)

project(FlexFlowTests)
set(project_target unit-test)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES False *.cc)

set(CPU_SRC
  ${TEST_SOURCES}
  )

add_custom_target(copy-test-makefile ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/graph_subst_3_v2.json)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/graph_subst_3_v2.json
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/substitutions/graph_subst_3_v2.json
                                                    ${CMAKE_CURRENT_BINARY_DIR}/graph_subst_3_v2.json
                   DEPENDS ${CMAKE_SOURCE_DIR}/substitutions/graph_subst_3_v2.json)

cuda_add_executable(${project_target} ${CPU_SRC})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR} ${gtest_SOURCE_DIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES} gtest gtest_main)
add_test(UnitTests ./${project_target})
