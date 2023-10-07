ff_add_library(
  NAME
    runtime
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    utils
    optional
    legion
    compiler
    kernels
    pcg
)

ff_add_test_executable(
  NAME
    runtime-test
  SRC_PATTERNS
    test/src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    test/src/ src/
  DEPS
    runtime
    doctest
)

add_subdirectory(ffi)
