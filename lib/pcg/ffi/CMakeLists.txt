set(target pcg-ffi)
set(internal_target pcg-ffi-internal)

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
    internal/
    src/
)
target_link_libraries(
  ${target}
  PRIVATE
    pcg
    internal-utils-ffi
    internal-op-attrs-ffi
  PUBLIC
    utils-ffi
    op-attrs-ffi
)

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

define_ff_vars(${target})
ff_set_cxx_properties(${target})
