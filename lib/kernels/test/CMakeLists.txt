set(project_target kernel-test)
project(${project_target} LANGUAGES CUDA)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc)

add_executable(
  ${project_target}
  ${SRC})

target_link_libraries(
  ${project_target}
  kernels
  cuda
  rapidcheck
  octest)

set_target_properties(
  ${project_target} 
  PROPERTIES 
  CUDA_STANDARD 11
)

define_ff_vars(${project_target})
doctest_discover_tests(${project_target})