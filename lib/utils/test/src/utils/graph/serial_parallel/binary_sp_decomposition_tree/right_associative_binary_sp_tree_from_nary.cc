#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("right_associative_binary_sp_tree_from_nary("
            "SerialParallelDecomposition)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    SUBCASE("only serial") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          SerialSplit{n1, n2, n3},
      };

      BinarySPDecompositionTree result =
          right_associative_binary_sp_tree_from_nary(input);
      BinarySPDecompositionTree correct = BinarySPDecompositionTree{
          BinarySeriesSplit{
              BinarySPDecompositionTree{n1},
              BinarySPDecompositionTree{BinarySeriesSplit{
                  BinarySPDecompositionTree{n2},
                  BinarySPDecompositionTree{n3},
              }},
          },
      };

      CHECK(result == correct);
    }
  }
}
