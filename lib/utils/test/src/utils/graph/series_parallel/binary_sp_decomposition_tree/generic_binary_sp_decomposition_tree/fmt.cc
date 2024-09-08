#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/fmt.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt GenericBinarySPDecompositionTree<int>") {
    SUBCASE("leaf") {
      GenericBinarySPDecompositionTree<int> input = make_generic_binary_sp_leaf(5);

      std::string result = fmt::to_string(input);
      std::string correct = "<GenericBinarySPDecompositionTree 5>";

      CHECK(result == correct);
    }

    SUBCASE("series split") {
      GenericBinarySPDecompositionTree<int> input = 
        make_generic_binary_series_split(
          make_generic_binary_sp_leaf(5),
          make_generic_binary_sp_leaf(7));

      std::string result = fmt::to_string(input);
      std::string correct = (
        "<GenericBinarySPDecompositionTree "
          "<GenericBinarySeriesSplit "
            "<GenericBinarySPDecompositionTree 5> "
            "<GenericBinarySPDecompositionTree 7>"
          ">"
        ">"
      );

      CHECK(result == correct);
    }

    SUBCASE("parallel split") {
      GenericBinarySPDecompositionTree<int> input = 
        make_generic_binary_parallel_split(
          make_generic_binary_sp_leaf(5),
          make_generic_binary_sp_leaf(7));

      std::string result = fmt::to_string(input);
      std::string correct = (
        "<GenericBinarySPDecompositionTree "
          "<GenericBinaryParallelSplit "
            "<GenericBinarySPDecompositionTree 5> "
            "<GenericBinarySPDecompositionTree 7>"
          ">"
        ">"
      );

      CHECK(result == correct);
    }
  }
}
