#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_binary_sp_tree_left_associative") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};

    GenericBinarySPDecompositionTreeImplementation<
      BinarySPDecompositionTree,
      BinarySeriesSplit,
      BinaryParallelSplit,
      Node> impl = generic_impl_for_binary_sp_tree();

    auto make_series_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) {
      return BinarySPDecompositionTree{n};
    };

    SUBCASE("input is actually left associative") {
      SUBCASE("just node") {
        BinarySPDecompositionTree input = make_leaf(n1);

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just series") {
        BinarySPDecompositionTree input = make_series_split(
            make_series_split(make_leaf(n1), make_leaf(n2)), make_leaf(n3));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        BinarySPDecompositionTree input = make_parallel_split(
            make_parallel_split(make_leaf(n1), make_leaf(n2)), make_leaf(n3));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("nested") {
        BinarySPDecompositionTree input = make_series_split(
            make_parallel_split(make_leaf(n1), make_leaf(n2)),
            make_parallel_split(make_leaf(n3), make_leaf(n4)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SUBCASE("input is not left associative") {
      SUBCASE("just series") {
        BinarySPDecompositionTree input = make_series_split(
            make_leaf(n1), make_series_split(make_leaf(n2), make_leaf(n3)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        BinarySPDecompositionTree input = make_parallel_split(
            make_leaf(n1), make_parallel_split(make_leaf(n2), make_leaf(n3)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }
    }
  }
}
