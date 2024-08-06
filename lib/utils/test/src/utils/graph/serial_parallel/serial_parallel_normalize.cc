#include "utils/graph/serial_parallel/serial_parallel_normalize.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("normalize function") {
    Node n1{1};
    Node n2{2};
    Node n3{3};

    SUBCASE("Node Decomposition") {
      SerialParallelDecomposition sp{n1};
      SerialParallelDecomposition expected = sp;
      CHECK(expected == normalize(sp));
    }

    SUBCASE("Serial with Single Node") {
      SerialParallelDecomposition sp{SerialSplit{n1}};
      SerialParallelDecomposition expected{n1};
      CHECK(expected == normalize(sp));
    }

    SUBCASE("Parallel with Single Node") {
      SerialParallelDecomposition sp{ParallelSplit{n1}};
      SerialParallelDecomposition expected{n1};
      CHECK(expected == normalize(sp));
    }

    SUBCASE("Mixed Serial") {
      SerialParallelDecomposition sp{SerialSplit{ParallelSplit{n1}, n2}};
      SerialParallelDecomposition expected{SerialSplit{n1, n2}};
      CHECK(expected == normalize(sp));
    }

    SUBCASE("Mixed Parallel") {
      SerialParallelDecomposition sp{ParallelSplit{SerialSplit{n1}, n2}};
      SerialParallelDecomposition expected{ParallelSplit{n1, n2}};
      CHECK(expected == normalize(sp));
    }

    SUBCASE("Nested") {
      SerialParallelDecomposition sp{
          ParallelSplit{SerialSplit{ParallelSplit{n1, n2}}, n3, SerialSplit{}}};
      SerialParallelDecomposition expected{ParallelSplit{n1, n2, n3}};
      CHECK(expected == normalize(sp));
    }
  }
}
