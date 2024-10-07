#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_num_tree_nodes.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_num_tree_nodes") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    GenericBinarySPDecompositionTreeImplementation<
      BinarySPDecompositionTree,
      BinarySeriesSplit,
      BinaryParallelSplit,
      Node> impl = generic_impl_for_binary_sp_tree();

    auto make_series_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) {
      return BinarySPDecompositionTree{n};
    };

    auto generic_get_num_tree_nodes = [&](BinarySPDecompositionTree const &tree) {
      return get_num_tree_nodes(tree, impl);
    };

    SUBCASE("leaf") {
      BinarySPDecompositionTree input =
          make_leaf(n1);

      int result = generic_get_num_tree_nodes(input);
      int correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("series split") {
      SUBCASE("children are not the same") {
        BinarySPDecompositionTree input =
            make_series_split(make_leaf(n1), make_leaf(n2));

        int result = generic_get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        BinarySPDecompositionTree input =
            make_series_split(make_leaf(n1), make_leaf(n1));

        int result = generic_get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallel split") {
      SUBCASE("children are not the same") {
        BinarySPDecompositionTree input =
            make_parallel_split(make_leaf(n1), make_leaf(n2));

        int result = generic_get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        BinarySPDecompositionTree input =
            make_parallel_split(make_leaf(n1), make_leaf(n1));

        int result = generic_get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }
    }

    SUBCASE("nested") {
      BinarySPDecompositionTree input =
          make_parallel_split(
              make_series_split(
                  make_leaf(n1),
                  make_series_split(
                      make_leaf(n2),
                      make_leaf(n3))),
              make_parallel_split(
                  make_leaf(n2),
                  make_leaf(n1)));

      int result = generic_get_num_tree_nodes(input);
      int correct = 9;

      CHECK(result == correct);
    }
  }
}
