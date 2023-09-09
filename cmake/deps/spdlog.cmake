include(aliasing)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/spdlog)

alias_library(spdlog spdlog::spdlog)
