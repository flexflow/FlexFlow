include(aliasing)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/json)

alias_library(nlohmann nlohmann_json::nlohmann_json)
