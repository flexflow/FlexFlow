set(project_target op-attrs)
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
  PRIVATE
    src/
  PUBLIC
    include/
)
target_link_libraries(
  ${project_target}
  mpark_variant
  optional
  utils
  visit_struct
)

define_ff_vars(${project_target})
ff_set_cxx_properties(${project_target})

add_subdirectory(ffi)
