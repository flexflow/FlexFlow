include(aliasing)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/variant)

alias_library(variant mpark_variant)
