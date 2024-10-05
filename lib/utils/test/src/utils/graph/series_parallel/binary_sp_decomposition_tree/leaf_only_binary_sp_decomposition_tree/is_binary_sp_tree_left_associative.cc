#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_binary_sp_tree_left_associative") {
    int n1 = 1;
    int n2 = 2;
    int n3 = 3;
    int n4 = 4;

    auto make_leaf = [](int n) {
      return leaf_only_binary_sp_tree_make_leaf(n);
    };

    auto make_series_split =
        [](LeafOnlyBinarySPDecompositionTree<int> const &l,
           LeafOnlyBinarySPDecompositionTree<int> const &r) {
          return leaf_only_binary_sp_tree_make_series_split(l, r);
        };

    auto make_parallel_split =
        [](LeafOnlyBinarySPDecompositionTree<int> const &l,
           LeafOnlyBinarySPDecompositionTree<int> const &r) {
          return leaf_only_binary_sp_tree_make_parallel_split(l, r);
        };

    SUBCASE("input is actually left associative") {
      SUBCASE("just node") {
        LeafOnlyBinarySPDecompositionTree<int> input = make_leaf(n1);

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just series") {
        LeafOnlyBinarySPDecompositionTree<int> input = make_series_split(
            make_series_split(make_leaf(n1), make_leaf(n2)), make_leaf(n3));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        LeafOnlyBinarySPDecompositionTree<int> input = make_parallel_split(
            make_parallel_split(make_leaf(n1), make_leaf(n2)), make_leaf(n3));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("nested") {
        LeafOnlyBinarySPDecompositionTree<int> input = make_series_split(
            make_parallel_split(make_leaf(n1), make_leaf(n2)),
            make_parallel_split(make_leaf(n3), make_leaf(n4)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SUBCASE("input is not left associative") {
      SUBCASE("just series") {
        LeafOnlyBinarySPDecompositionTree<int> input = make_series_split(
            make_leaf(n1), make_series_split(make_leaf(n2), make_leaf(n3)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        LeafOnlyBinarySPDecompositionTree<int> input = make_parallel_split(
            make_leaf(n1), make_parallel_split(make_leaf(n2), make_leaf(n3)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }
    }
  }
}
