set(project_target pcg)
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
    include/
  INTERFACE
    include/
)
target_link_libraries(
  ${project_target}
  utils
  json
  optional
  visit_struct
  op-attrs
)

define_ff_vars(${project_target})
ff_set_cxx_properties(${project_target})

add_subdirectory(ffi)
