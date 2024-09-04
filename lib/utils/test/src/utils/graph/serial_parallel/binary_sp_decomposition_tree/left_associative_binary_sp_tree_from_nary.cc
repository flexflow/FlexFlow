#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/get_nodes.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include <doctest/doctest.h>
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/rapidcheck.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("left_associative_binary_sp_tree_from_nary(SerialParallelDecomposition)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    SUBCASE("only node") {
      SerialParallelDecomposition input = SerialParallelDecomposition{n1};

      BinarySPDecompositionTree result = left_associative_binary_sp_tree_from_nary(input);
      BinarySPDecompositionTree correct = BinarySPDecompositionTree::node(n1);

      CHECK(result == correct);
    }

    SUBCASE("only serial") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
        SerialSplit{n1, n2, n3},  
      };

      BinarySPDecompositionTree result = left_associative_binary_sp_tree_from_nary(input);
      BinarySPDecompositionTree correct = BinarySPDecompositionTree::series(
        BinarySPDecompositionTree::series(
          BinarySPDecompositionTree::node(n1),
          BinarySPDecompositionTree::node(n2)
        ),
        BinarySPDecompositionTree::node(n3)
      );

      CHECK(result == correct);
    }

    SUBCASE("only parallel") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
        ParallelSplit{n1, n2, n3},  
      };

      BinarySPDecompositionTree result = left_associative_binary_sp_tree_from_nary(input);

      CHECK(is_binary_sp_tree_left_associative(result));

      std::unordered_multiset<Node> result_nodes = get_nodes(result);
      std::unordered_multiset<Node> correct_nodes = get_nodes(input);

      CHECK(result_nodes == correct_nodes);
    }

    // TODO(@lockshaw) add rapidcheck support for SerialParallelDecomposition
    RC_SUBCASE([](BinarySPDecompositionTree const &binary) {
      SerialParallelDecomposition nary = nary_sp_tree_from_binary(binary);
      BinarySPDecompositionTree result = left_associative_binary_sp_tree_from_nary(nary);

      CHECK(is_binary_sp_tree_left_associative(result));

      std::unordered_multiset<Node> result_nodes = get_nodes(result);
      std::unordered_multiset<Node> correct_nodes = get_nodes(nary);

      CHECK(result_nodes == correct_nodes);
    });
  }
}
