#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/transform.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform(GenericBinarySPDecompositionTree<int>, F)") {
    FAIL("TODO");
    // GenericBinarySPDecompositionTree<int> input =
    //     make_generic_binary_parallel_split(
    //         make_generic_binary_series_split(make_generic_binary_sp_leaf(1),
    //                                          make_generic_binary_sp_leaf(4)),
    //         make_generic_binary_sp_leaf(2));
    //
    // GenericBinarySPDecompositionTree<std::string> result =
    //     transform(input, [](int x) { return std::to_string(x); });
    //
    // GenericBinarySPDecompositionTree<std::string> correct =
    //     make_generic_binary_parallel_split(
    //         make_generic_binary_series_split(
    //             make_generic_binary_sp_leaf(std::string{"1"}),
    //             make_generic_binary_sp_leaf(std::string{"4"})),
    //         make_generic_binary_sp_leaf(std::string{"2"}));
    //
    // CHECK(result == correct);
  }
}
