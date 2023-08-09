set(project_target runtime)
set(test_target runtime-test)

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
  op-attrs
  utils
  optional
  legion
  compiler
  kernels
  pcg
)

define_ff_vars(${project_target})
ff_set_cxx_properties(${project_target})

project(${test_target})

file(GLOB_RECURSE TEST_SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     test/*.cc)

add_executable(${test_target}
  ${TEST_SRC})

target_include_directories(
  ${test_target}
  PRIVATE
    src/ test/
  PUBLIC
    include/
)
target_link_libraries(
  ${test_target}
  ${project_target}
  doctest
)
doctest_discover_tests(${test_target})
ff_set_cxx_properties(${test_target})

add_subdirectory(ffi)
