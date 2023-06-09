include(aliasing)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/invoke)

alias_library(invoke invoke.hpp::invoke.hpp)
