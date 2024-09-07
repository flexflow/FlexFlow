#include "utils/graph/serial_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Arbitrary<BinarySPDecompositionTree>") {
    FAIL("TODO");
    // for (int i = 0; i < 100; i++) {
    //   BinarySPDecompositionTree generated =
    //       *rc::gen::resize(20, rc::gen::arbitrary<BinarySPDecompositionTree>());
    //
    //   int num_tree_nodes = get_num_tree_nodes(generated);
    //
    //   CHECK(num_tree_nodes > 50);
    //   CHECK(num_tree_nodes <= 100);
    // }
  }
}
