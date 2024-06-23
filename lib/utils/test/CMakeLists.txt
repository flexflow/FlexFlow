ff_add_test_executable(
  NAME
    utils-tests
  SRC_PATTERNS
    src/test_cow_ptr.cc
    src/utils/graph/dataflow_graph/unordered_set_dataflow_graph.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    doctest
    utils-test-common
)

add_subdirectory(common)
