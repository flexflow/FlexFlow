ff_add_test_executable(
  NAME
    utils-tests
  SRC_PATTERNS
    src/*
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    # doctest
    utils-test-common
)

add_subdirectory(common)
