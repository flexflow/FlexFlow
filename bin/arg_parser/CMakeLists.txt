cmake_minimum_required(VERSION 3.10)

project(arg_parser)
set(project_target arg_parser)

add_executable(${project_target} arg_parser.cc)
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive ${FLEXFLOW_LIBS} -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
