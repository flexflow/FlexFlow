#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "utils/json.h"
#include <doctest/doctest.h>
#include <iostream>
#include <sstream>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BatchNormAttrs to/from json") {
    BatchNormAttrs correct = BatchNormAttrs{true};
    json j = correct;
    auto result = j.get<BatchNormAttrs>();
    CHECK(result == correct);
  }

  TEST_CASE("ComputationGraphAttrs to/from json") {
    ComputationGraphOpAttrs correct =
        ComputationGraphOpAttrs{BatchNormAttrs{true}};
    json j = correct;
    auto result = j.get<ComputationGraphOpAttrs>();

    CHECK(result == correct);
  }

  TEST_CASE("PCGOperatorAttrs to/from json") {
    PCGOperatorAttrs correct = PCGOperatorAttrs{RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1},
        /*repartition_degree=*/4,
    }};
    json j = correct;
    auto result = j.get<PCGOperatorAttrs>();

    CHECK(result == correct);
  }
}
