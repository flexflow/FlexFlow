#include "utils/graph/series_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("nary_sp_tree_from_binary(BinarySPDecompositionTree)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};
    Node n5 = Node{5};
    Node n6 = Node{6};

    auto make_series_split = [](BinarySPDecompositionTree const &lhs,
                                BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs,
                                  BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) { return BinarySPDecompositionTree{n}; };

    SUBCASE("leaf") {
      BinarySPDecompositionTree input = make_leaf(n1);

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};

      CHECK(result == correct);
    }

    SUBCASE("left associative series") {
      BinarySPDecompositionTree input = make_series_split(
          make_series_split(make_leaf(n2), make_leaf(n1)), make_leaf(n3));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n2, n1, n3}}};

      CHECK(result == correct);
    }

    SUBCASE("right associative series") {
      BinarySPDecompositionTree input = make_series_split(
          make_leaf(n2), make_series_split(make_leaf(n1), make_leaf(n3)));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n2, n1, n3}}};

      CHECK(result == correct);
    }

    SUBCASE("series with duplicate children") {
      BinarySPDecompositionTree input =
          make_series_split(make_leaf(n1), make_leaf(n1));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n1, n1}}};

      CHECK(get_nodes(result).size() == 2);
      CHECK(result == correct);
    }

    SUBCASE("left associative parallel") {
      BinarySPDecompositionTree input = make_parallel_split(
          make_parallel_split(make_leaf(n2), make_leaf(n1)), make_leaf(n3));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n2, n1, n3}}};

      CHECK(result == correct);
    }

    SUBCASE("right associative parallel") {
      BinarySPDecompositionTree input = make_parallel_split(
          make_leaf(n2), make_parallel_split(make_leaf(n1), make_leaf(n3)));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n2, n1, n3}}};

      CHECK(result == correct);
    }

    SUBCASE("parallel with duplicate children") {
      BinarySPDecompositionTree input =
          make_parallel_split(make_leaf(n1), make_leaf(n1));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n1, n1}}};

      CHECK(get_nodes(result).size() == 2);
      CHECK(result == correct);
    }

    SUBCASE("nested") {
      BinarySPDecompositionTree input = make_parallel_split(
          make_parallel_split(
              make_parallel_split(
                  make_leaf(n1),
                  make_series_split(
                      make_series_split(
                          make_series_split(make_leaf(n2), make_leaf(n3)),
                          make_leaf(n3)),
                      make_leaf(n5))),
              make_series_split(make_leaf(n6), make_leaf(n4))),
          make_leaf(n5));

      SeriesParallelDecomposition result = nary_sp_tree_from_binary(input);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
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

      CHECK(result == correct);
    }
  }
}
