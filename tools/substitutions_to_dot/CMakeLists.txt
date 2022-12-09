cmake_minimum_required(VERSION 3.6)

include(json)

project(visualizeTool)
set(project_target visualize)


add_executable(${project_target} substitution_to_dot.cc)
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES} nlohmann_json::nlohmann_json substitution_loader)
