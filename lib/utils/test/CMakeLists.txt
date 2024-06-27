ff_add_test_executable(
  NAME
    utils-tests
  SRC_PATTERNS
    src/test_cow_ptr.cc
    src/test_optional.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    doctest
    utils-test-common
)

add_subdirectory(common)
