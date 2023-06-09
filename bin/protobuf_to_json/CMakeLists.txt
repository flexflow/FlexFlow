cmake_minimum_required(VERSION 3.6)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_SRC PROTO_HEADER rules.proto)
add_library(proto ${PROTO_HEADER} ${PROTO_SRC})

include(json)

project(FlexFlow_substitutionTool)
set(project_target substitution_tool)

add_executable(${project_target} protobuf_to_json.cc)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${project_target} proto ${PROTOBUF_LIBRARY} nlohmann_json::nlohmann_json)
# target_link_libraries(${project_target} PRIVATE nlohmann_json::nlohmann_json)
