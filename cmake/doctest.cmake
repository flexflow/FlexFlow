include(aliasing)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest)
include(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest/scripts/cmake/doctest.cmake)

alias_library(doctest doctest::doctest)
