ff_add_test_executable(
  NAME
    substitutions-test
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    doctest
    rapidcheck
    substitutions
)

