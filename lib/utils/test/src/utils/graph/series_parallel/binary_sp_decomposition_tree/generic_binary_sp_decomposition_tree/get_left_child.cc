#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/fmt.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_left_child(GenericBinarySPDecompositionTree<T>)") {
    SUBCASE("leaf") {
      GenericBinarySPDecompositionTree<int> input =
          make_generic_binary_sp_leaf(5);

      CHECK_THROWS(get_left_child(input));
    }

    SUBCASE("series split") {
      GenericBinarySPDecompositionTree<int> input =
          make_generic_binary_series_split(make_generic_binary_sp_leaf(5),
                                           make_generic_binary_sp_leaf(3));

      GenericBinarySPDecompositionTree<int> result = get_left_child(input);
      GenericBinarySPDecompositionTree<int> correct =
          make_generic_binary_sp_leaf(5);

      CHECK(result == correct);
    }

    SUBCASE("parallel split") {
      GenericBinarySPDecompositionTree<int> input =
          make_generic_binary_parallel_split(make_generic_binary_sp_leaf(4),
                                             make_generic_binary_sp_leaf(7));

      GenericBinarySPDecompositionTree<int> result = get_left_child(input);
      GenericBinarySPDecompositionTree<int> correct =
          make_generic_binary_sp_leaf(4);

      CHECK(result == correct);
    }
  }
}
