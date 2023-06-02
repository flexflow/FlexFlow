set(target substitutions-ffi)

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
    utils-ffi
  PRIVATE
    substitutions
)

ff_set_cxx_properties(${target})
