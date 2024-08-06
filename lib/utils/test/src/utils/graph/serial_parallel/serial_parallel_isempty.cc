#include "utils/graph/serial_parallel/serial_parallel_isempty.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("isempty function") {
    Node n1{1};
    Node n2{2};

    SUBCASE("Node Decomposition") {
      SerialParallelDecomposition sp{n1};
      CHECK_FALSE(isempty(sp));
    }

    SUBCASE("Empty Serial") {
      SerialParallelDecomposition sp{SerialSplit{}};
      CHECK(isempty(sp));
    }

    SUBCASE("Empty Parallel") {
      SerialParallelDecomposition sp{ParallelSplit{}};
      CHECK(isempty(sp));
    }

    SUBCASE("Serial with Node") {
      SerialParallelDecomposition sp{SerialSplit{n1}};
      CHECK_FALSE(isempty(sp));
    }

    SUBCASE("Parallel with Node") {
      SerialParallelDecomposition sp{ParallelSplit{n1}};
      CHECK_FALSE(isempty(sp));
    }

    SUBCASE("Nested Serial") {
      SerialParallelDecomposition sp{SerialSplit{ParallelSplit{}}};
      CHECK(isempty(sp));
    }

    SUBCASE("Nested Parallel") {
      SerialParallelDecomposition sp{ParallelSplit{SerialSplit{}}};
      CHECK(isempty(sp));
    }

    SUBCASE("Sparse") {
      SerialSplit sp{ParallelSplit{}, ParallelSplit{SerialSplit{}}};
      CHECK(isempty(sp));
    }

    SUBCASE("Sparse with Node") {
      SerialSplit sp{ParallelSplit{}, ParallelSplit{SerialSplit{}, n2}};
      CHECK_FALSE(isempty(sp));
    }
  }
}
