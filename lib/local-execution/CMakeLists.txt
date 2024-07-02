ff_add_library(
  NAME
    local-execution
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
    spdlog
)

add_subdirectory(test)
