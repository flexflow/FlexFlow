ff_add_test_executable(
  NAME
    op-attrs-test
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    doctest
    op-attrs-test-common
)

add_subdirectory(common)
