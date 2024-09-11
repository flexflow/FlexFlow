ff_add_library(
  NAME
    substitution-generator
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    op-attrs
    pcg
    substitutions
)

# add_subdirectory(ffi)
add_subdirectory(test)
