include(aliasing)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fmt)

alias_library(fmt fmt::fmt)
