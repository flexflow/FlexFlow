set(target runtime-ffi)
set(internal_target runtime-ffi-internal)

project(${target})
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
  PUBLIC
    include/
  PRIVATE
    src/
)
target_link_libraries(
  ${target}
  PUBLIC
    compiler-ffi
    internal-utils-ffi
    substitutions-ffi
    pcg-ffi
    op-attrs-ffi
  PRIVATE
    runtime
    utils
)

ff_set_cxx_properties(${target})

add_library(
  ${internal_target}
  INTERFACE
)
target_link_libraries(
  ${internal_target}
  INTERFACE 
    ${target}
)
target_include_directories(
  ${internal_target}
  INTERFACE
    internal/
)

