ff_add_test_executable(
  NAME
    utils-test
  SRC_PATTERNS
    src/test_cow_ptr.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    doctest
    utils-test-common
)

add_subdirectory(common)
