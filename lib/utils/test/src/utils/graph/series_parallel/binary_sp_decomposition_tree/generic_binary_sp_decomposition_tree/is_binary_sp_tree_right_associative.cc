#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/leaf_only_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_binary_sp_tree_right_associative("
            "LeafOnlyBinarySPDecompositionTree<int>)") {
    FAIL("TODO");
    // int n1 = 1;
    // int n2 = 2;
    // int n3 = 3;
    // int n4 = 4;
    //
    // SUBCASE("input is actually right associative") {
    //   SUBCASE("just node") {
    //     LeafOnlyBinarySPDecompositionTree<int> input =
    //         make_generic_binary_sp_leaf(n1);
    //
    //     bool result = is_binary_sp_tree_right_associative(input);
    //     bool correct = true;
    //
    //     CHECK(result == correct);
    //   }
    //
    //   SUBCASE("just series") {
    //     LeafOnlyBinarySPDecompositionTree<int> input =
    //         make_generic_binary_series_split(
    //             make_generic_binary_sp_leaf(n1),
    //             make_generic_binary_series_split(
    //                 make_generic_binary_sp_leaf(n2),
    //                 make_generic_binary_sp_leaf(n3)));
    //
    //     bool result = is_binary_sp_tree_right_associative(input);
    //     bool correct = true;
    //
    //     CHECK(result == correct);
    //   }
    //
    //   SUBCASE("just parallel") {
    //     LeafOnlyBinarySPDecompositionTree<int> input =
    //         make_generic_binary_parallel_split(
    //             make_generic_binary_sp_leaf(n1),
    //             make_generic_binary_parallel_split(
    //                 make_generic_binary_sp_leaf(n2),
    //                 make_generic_binary_sp_leaf(n3)));
    //
    //     bool result = is_binary_sp_tree_right_associative(input);
    //     bool correct = true;
    //
    //     CHECK(result == correct);
    //   }
    //
    //   SUBCASE("nested") {
    //     LeafOnlyBinarySPDecompositionTree<int> input =
    //         make_generic_binary_series_split(
    //             make_generic_binary_parallel_split(
    //                 make_generic_binary_sp_leaf(n1),
    //                 make_generic_binary_sp_leaf(n2)),
    //             make_generic_binary_parallel_split(
    //                 make_generic_binary_sp_leaf(n3),
    //                 make_generic_binary_sp_leaf(n4)));
    //
    //     bool result = is_binary_sp_tree_right_associative(input);
    //     bool correct = true;
    //
    //     CHECK(result == correct);
    //   }
    // }
    //
    // SUBCASE("input is not right associative") {
    //   SUBCASE("just series") {
    //     LeafOnlyBinarySPDecompositionTree<int> input =
    //         make_generic_binary_series_split(
    //             make_generic_binary_series_split(
    //                 make_generic_binary_sp_leaf(n1),
    //                 make_generic_binary_sp_leaf(n2)),
    //             make_generic_binary_sp_leaf(n3));
    //
    //     bool result = is_binary_sp_tree_right_associative(input);
    //     bool correct = false;
    //
    //     CHECK(result == correct);
    //   }
    //
    //   SUBCASE("just parallel") {
    //     LeafOnlyBinarySPDecompositionTree<int> input =
    //         make_generic_binary_parallel_split(
    //             make_generic_binary_parallel_split(
    //                 make_generic_binary_sp_leaf(n1),
    //                 make_generic_binary_sp_leaf(n2)),
    //             make_generic_binary_sp_leaf(n3));
    //
    //     bool result = is_binary_sp_tree_right_associative(input);
    //     bool correct = false;
    //
    //     CHECK(result == correct);
    //   }
    // }
  }
}
