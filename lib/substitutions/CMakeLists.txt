ff_add_library(
  NAME
    substitutions
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    op-attrs
)

add_subdirectory(ffi)
# add_subdirectory(test)
