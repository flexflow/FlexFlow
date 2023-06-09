set(target utils-ffi)

project(${target})
file(GLOB_RECURSE FFI_SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc)
add_library(
  ${target}
  SHARED
  ${FFI_SRC}
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
  PRIVATE
    utils
)

ff_set_cxx_properties(${target})
