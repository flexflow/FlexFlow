#include "utils/graph/serial_parallel/normalize_sp_decomposition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("normalize_sp_decomposition") {
    Node n1 = Node(1);
    Node n2 = Node(2);
    Node n3 = Node(3);

    SUBCASE("Empty") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          SerialSplit{ParallelSplit{}, ParallelSplit{}}};
      SerialParallelDecomposition correct =
          SerialParallelDecomposition{SerialSplit{}};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Node Decomposition") {
      SerialParallelDecomposition input = SerialParallelDecomposition{n1};
      SerialParallelDecomposition correct = SerialParallelDecomposition{n1};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Serial with Single Node") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{SerialSplit{n1}};
      SerialParallelDecomposition correct = SerialParallelDecomposition{n1};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Parallel with Single Node") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{ParallelSplit{n1}};
      SerialParallelDecomposition correct = SerialParallelDecomposition{n1};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Mixed Serial") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{SerialSplit{ParallelSplit{n1}, n2}};
      SerialParallelDecomposition correct =
          SerialParallelDecomposition{SerialSplit{n1, n2}};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Mixed Parallel") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{ParallelSplit{SerialSplit{n1}, n2}};
      SerialParallelDecomposition correct =
          SerialParallelDecomposition{ParallelSplit{n1, n2}};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Nested") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          ParallelSplit{SerialSplit{ParallelSplit{n1, n2}}, n3, SerialSplit{}}};
      SerialParallelDecomposition correct =
          SerialParallelDecomposition{ParallelSplit{n1, n2, n3}};
      SerialParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }
  }
}
