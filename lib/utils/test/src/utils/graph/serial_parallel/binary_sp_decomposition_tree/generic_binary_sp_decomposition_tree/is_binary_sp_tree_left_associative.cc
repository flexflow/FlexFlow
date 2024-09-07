#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/fmt.h"
#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_binary_sp_tree_left_associative("
            "GenericBinarySPDecompositionTree<int>)") {
    int n1 = 1;
    int n2 = 2;
    int n3 = 3;
    int n4 = 4;

    SUBCASE("input is actually left associative") {
      SUBCASE("just node") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_sp_leaf(n1);

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just series") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_series_split(
                make_generic_binary_series_split(
                    make_generic_binary_sp_leaf(n1),
                    make_generic_binary_sp_leaf(n2)),
                make_generic_binary_sp_leaf(n3));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_parallel_split(
                make_generic_binary_parallel_split(
                    make_generic_binary_sp_leaf(n1),
                    make_generic_binary_sp_leaf(n2)),
                make_generic_binary_sp_leaf(n3));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("nested") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_series_split(
                make_generic_binary_parallel_split(
                    make_generic_binary_sp_leaf(n1),
                    make_generic_binary_sp_leaf(n2)),
                make_generic_binary_parallel_split(
                    make_generic_binary_sp_leaf(n3),
                    make_generic_binary_sp_leaf(n4)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SUBCASE("input is not left associative") {
      SUBCASE("just series") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_series_split(
                make_generic_binary_sp_leaf(n1),
                make_generic_binary_series_split(
                    make_generic_binary_sp_leaf(n2),
                    make_generic_binary_sp_leaf(n3)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        GenericBinarySPDecompositionTree<int> input =
            make_generic_binary_parallel_split(
                make_generic_binary_sp_leaf(n1),
                make_generic_binary_parallel_split(
                    make_generic_binary_sp_leaf(n2),
                    make_generic_binary_sp_leaf(n3)));

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }
    }
  }
}
