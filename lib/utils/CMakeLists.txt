set(project_target utils)
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
set_target_properties(
  ${project_target} 
  PROPERTIES 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

add_subdirectory(test)
