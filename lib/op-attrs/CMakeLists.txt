ff_add_library(
  NAME
    op-attrs
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/ 
  DEPS
    utils
)

add_subdirectory(ffi)
