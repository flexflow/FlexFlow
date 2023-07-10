set(target compiler-ffi)
project(${target} CXX)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc)
add_library(
  ${target}
  SHARED
  ${SRC}
)
target_include_directories(
  ${target}
  PRIVATE
    src/
  PUBLIC
    include/
)
target_link_libraries(
  ${target}
  PUBLIC
    utils-ffi
    pcg-ffi
  PRIVATE
    compiler
)
ff_set_cxx_properties(${target})
