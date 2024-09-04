#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/is_binary_sp_tree_left_associative.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_binary_sp_tree_left_associative(BinarySPDecompositionTree)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};

    SUBCASE("input is actually left associative") {
      SUBCASE("just node") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree::node(n1);

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just series") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree::series(
          BinarySPDecompositionTree::series(
            BinarySPDecompositionTree::node(n1),
            BinarySPDecompositionTree::node(n2)
          ),
          BinarySPDecompositionTree::node(n3)
        );

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree::parallel(
          BinarySPDecompositionTree::parallel(
            BinarySPDecompositionTree::node(n1),
            BinarySPDecompositionTree::node(n2)
          ),
          BinarySPDecompositionTree::node(n3)
        );

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("nested") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree::series(
          BinarySPDecompositionTree::parallel(
            BinarySPDecompositionTree::node(n1),
            BinarySPDecompositionTree::node(n2)
          ),
          BinarySPDecompositionTree::parallel(
            BinarySPDecompositionTree::node(n3),
            BinarySPDecompositionTree::node(n4)
          )
        );

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SUBCASE("input is not left associative") {
      SUBCASE("just series") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree::series(
          BinarySPDecompositionTree::node(n1),
          BinarySPDecompositionTree::series(
            BinarySPDecompositionTree::node(n2),
            BinarySPDecompositionTree::node(n3)
          )
        );

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }

      SUBCASE("just parallel") {
        BinarySPDecompositionTree input = BinarySPDecompositionTree::parallel(
          BinarySPDecompositionTree::node(n1),
          BinarySPDecompositionTree::parallel(
            BinarySPDecompositionTree::node(n2),
            BinarySPDecompositionTree::node(n3)
          )
        );

        bool result = is_binary_sp_tree_left_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }
    }
  }
}
