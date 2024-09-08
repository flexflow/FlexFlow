#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("right_associative_binary_sp_tree_from_nary("
            "SeriesParallelDecomposition)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    SUBCASE("only serial") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{n1, n2, n3},
      };

      BinarySPDecompositionTree result =
          right_associative_binary_sp_tree_from_nary(input);
      BinarySPDecompositionTree correct = make_series_split(
          make_leaf_node(n1),
          make_series_split(make_leaf_node(n2), make_leaf_node(n3)));

      CHECK(result == correct);
    }
  }
}
