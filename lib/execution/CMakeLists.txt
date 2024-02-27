ff_add_library(
  NAME
    execution
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    utils
    kernels
    pcg
)

# ff_add_test_executable(
#   NAME
#     execution-test
#   SRC_PATTERNS
#     test/src/*.cc
#   PUBLIC_INCLUDE
#     include/
#   PRIVATE_INCLUDE
#     test/src/ src/
#   DEPS
#     execution
#     doctest
# )

# add_subdirectory(ffi)
