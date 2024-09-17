#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ComputationGraphOpAttrs to/from json") {
    ComputationGraphOpAttrs correct =
        ComputationGraphOpAttrs{BatchNormAttrs{true}};
    nlohmann::json j = correct;
    auto result = j.get<ComputationGraphOpAttrs>();

    CHECK(result == correct);
  }
}
