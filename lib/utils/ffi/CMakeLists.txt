set(target utils-ffi)
set(internal_target "internal-${target}")

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
    internal/
)
target_link_libraries(
  ${target}
  PRIVATE
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
