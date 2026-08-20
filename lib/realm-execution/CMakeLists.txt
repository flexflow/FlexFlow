project(realm-execution LANGUAGES CXX CUDA)

ff_add_library(
  NAME
    realm-execution
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    compiler
    kernels
    local-execution
    op-attrs
    pcg
    deps::spdlog
    task-spec
    utils
    deps::realm
)

# redops/realm_redop_registry.cc needs to be compiled by nvcc (not g++) so
# that its GPU reduction kernels (SumReduction<T>::apply_cuda/fold_cuda)
# actually get generated; see the comment on register_sum_redop for why
# that matters. Kept as a .cc file (rather than renamed to .cu, as lib/
# kernels does for its own CUDA sources) so it stays subject to proj's
# normal .h/.cc file-group layout check — realm-execution isn't in
# .proj.toml's layout_ignore_paths the way lib/kernels is.
set_source_files_properties(
  src/realm-execution/redops/realm_redop_registry.cc
  PROPERTIES
  LANGUAGE CUDA
)

set_target_properties(
  realm-execution
  PROPERTIES
  CUDA_STANDARD 17
)

add_subdirectory(test)
