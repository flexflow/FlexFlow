#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_num_tree_nodes.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_num_tree_nodes(GenericBinarySPDecompositionTree<int>)") {
    SUBCASE("leaf") {
      GenericBinarySPDecompositionTree<int> input =
          make_generic_binary_sp_leaf(5);

      int result = get_num_tree_nodes(input);
      int correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("series split") {
      SUBCASE("children are not the same") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_series_split(make_generic_binary_sp_leaf(5),
                                             make_generic_binary_sp_leaf(6));

        int result = get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_series_split(make_generic_binary_sp_leaf(5),
                                             make_generic_binary_sp_leaf(5));

        int result = get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallel split") {
      SUBCASE("children are not the same") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_parallel_split(make_generic_binary_sp_leaf(5),
                                               make_generic_binary_sp_leaf(6));

        int result = get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_parallel_split(make_generic_binary_sp_leaf(5),
                                               make_generic_binary_sp_leaf(5));

        int result = get_num_tree_nodes(input);
        int correct = 3;

        CHECK(result == correct);
      }
    }

    SUBCASE("nested") {
      GenericBinarySPDecompositionTree<int> input =
          make_generic_binary_parallel_split(
              make_generic_binary_series_split(
                  make_generic_binary_sp_leaf(4),
                  make_generic_binary_series_split(
                      make_generic_binary_sp_leaf(2),
                      make_generic_binary_sp_leaf(5))),
              make_generic_binary_parallel_split(
                  make_generic_binary_sp_leaf(4),
                  make_generic_binary_sp_leaf(2)));

      int result = get_num_tree_nodes(input);
      int correct = 9;

      CHECK(result == correct);
    }
  }
}
