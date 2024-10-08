#include "utils/graph/series_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/rapidcheck.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("left_associative_binary_sp_tree_from_nary("
            "SeriesParallelDecomposition)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};
    Node n5 = Node{5};
    Node n6 = Node{6};

    auto make_series_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) {
      return BinarySPDecompositionTree{n};
    };

    SUBCASE("only node") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{n1};

      BinarySPDecompositionTree result =
          left_associative_binary_sp_tree_from_nary(input);
      BinarySPDecompositionTree correct = make_leaf(n1);

      CHECK(result == correct);
    }

    SUBCASE("only serial") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{n1, n2, n3}},
      };

      BinarySPDecompositionTree result =
          left_associative_binary_sp_tree_from_nary(input);

      BinarySPDecompositionTree correct = make_series_split(
          make_series_split(make_leaf(n1), make_leaf(n2)),
          make_leaf(n3));

      CHECK(result == correct);
    }

    SUBCASE("only parallel") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{n1, n2, n3}},
      };

      BinarySPDecompositionTree result =
          left_associative_binary_sp_tree_from_nary(input);

      // we use multiple checks here because SerialParallelDecomposition's
      // ParallelSplit is unordered, so there are multiple possible
      // left-associative binary SP trees
      CHECK(is_binary_sp_tree_left_associative(result));

      std::unordered_multiset<Node> result_nodes = get_nodes(input);
      std::unordered_multiset<Node> correct_nodes = {n1, n2, n3};

      CHECK(result_nodes == correct_nodes);
    }

    SUBCASE("nested") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{
              n1,
              SeriesSplit{{
                  n2,
                  n3,
                  n3,
                  n5,
              }},
              SeriesSplit{{
                  n6,
                  n4,
              }},
              n5,
          }},
      };

      BinarySPDecompositionTree result =
          left_associative_binary_sp_tree_from_nary(input);

      CHECK(is_binary_sp_tree_left_associative(result));

      std::unordered_multiset<Node> result_nodes = get_nodes(input);
      std::unordered_multiset<Node> correct_nodes = {
          n1, n2, n3, n3, n5, n6, n4, n5};

      CHECK(result_nodes == correct_nodes);
    }
  }
}
