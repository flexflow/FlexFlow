set(test_target pcg-test)
project(${test_target})

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     *.cc)

add_executable(
  ${test_target}
  ${SRC})

define_ff_vars(${test_target})

target_link_libraries(
  ${test_target}
  pcg
  utils
  doctest::doctest)

set_target_properties(
  ${test_target}
  PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
doctest_discover_tests(${test_target})
