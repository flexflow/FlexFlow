#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_leaves.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_leaves(GenericBinarySPDecompositionTree<int>)") {
    SUBCASE("leaf") {
      GenericBinarySPDecompositionTree<int> input = make_generic_binary_sp_leaf(5);

      std::unordered_multiset<int> result = get_leaves(input);
      std::unordered_multiset<int> correct = {5};

      CHECK(result == correct);
    }

    SUBCASE("series split") {
      SUBCASE("children are not the same") {
        GenericBinarySPDecompositionTree<int> input = 
          make_generic_binary_series_split(
            make_generic_binary_sp_leaf(5),
            make_generic_binary_sp_leaf(6));

        std::unordered_multiset<int> result = get_leaves(input);
        std::unordered_multiset<int> correct = {5, 6};

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        GenericBinarySPDecompositionTree<int> input = 
          make_generic_binary_series_split(
            make_generic_binary_sp_leaf(5),
            make_generic_binary_sp_leaf(5));

        std::unordered_multiset<int> result = get_leaves(input);
        std::unordered_multiset<int> correct = {5, 5};

        CHECK(result == correct);
      }
    }

    SUBCASE("parallel split") {
      SUBCASE("children are not the same") {
        GenericBinarySPDecompositionTree<int> input = 
          make_generic_binary_parallel_split(
            make_generic_binary_sp_leaf(5),
            make_generic_binary_sp_leaf(6));

        std::unordered_multiset<int> result = get_leaves(input);
        std::unordered_multiset<int> correct = {5, 6};

        CHECK(result == correct);
      }

      SUBCASE("children are the same") {
        GenericBinarySPDecompositionTree<int> input = 
          make_generic_binary_parallel_split(
            make_generic_binary_sp_leaf(5),
            make_generic_binary_sp_leaf(5));

        std::unordered_multiset<int> result = get_leaves(input);
        std::unordered_multiset<int> correct = {5, 5};

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

      std::unordered_multiset<int> result = get_leaves(input);
      std::unordered_multiset<int> correct = {2, 2, 4, 4, 5};

      CHECK(result == correct);
    }
  }
}
