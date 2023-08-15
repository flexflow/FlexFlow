set(project_target kernel-test)
project(${project_target})

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
  doctest::doctest)
  
set_target_properties(
  ${project_target} 
  PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
doctest_discover_tests(${project_target})
