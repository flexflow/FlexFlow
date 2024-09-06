#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <doctest/doctest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BatchNormAttrs to/from json") {
    BatchNormAttrs correct = BatchNormAttrs{true};
    nlohmann::json j = correct;
    BatchNormAttrs result = j.get<BatchNormAttrs>();
    CHECK(result == correct);
  }

  TEST_CASE("ComputationGraphAttrs to/from json") {
    ComputationGraphOpAttrs correct =
        ComputationGraphOpAttrs{BatchNormAttrs{true}};
    nlohmann::json j = correct;
    ComputationGraphOpAttrs result = j.get<ComputationGraphOpAttrs>();

    CHECK(result == correct);
  }

  TEST_CASE("PCGOperatorAttrs to/from json") {
    PCGOperatorAttrs correct = PCGOperatorAttrs{RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1},
        /*repartition_degree=*/4,
    }};
    nlohmann::json j = correct;
    PCGOperatorAttrs result = j.get<PCGOperatorAttrs>();

    CHECK(result == correct);
  }
}
