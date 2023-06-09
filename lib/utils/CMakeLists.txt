set(project_target utils)
set(ffi_target ${project_target}-ffi)
project(${project_target})

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc)

add_library(
  ${project_target}
  SHARED
  ${SRC}
)
target_include_directories(
  ${project_target}
  PUBLIC
    include/
  PRIVATE
    src/
)
target_link_libraries(
  ${project_target}
  optional
  expected
  mpark_variant
  visit_struct
  fmt::fmt
  rapidcheck
)
ff_set_cxx_properties(${project_target})

add_subdirectory(ffi)
add_subdirectory(test)
