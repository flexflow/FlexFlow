#include "doctest/doctest.h"
#include "op-attrs/operator_attrs.h"
#include "utils/json.h"
#include <sstream>
#include <iostream>

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BatchNormAttrs to/from json") {
    BatchNormAttrs correct = BatchNormAttrs{true};
    json j = correct;
    auto result = j.get<BatchNormAttrs>();
    CHECK(result == correct);
  }

  TEST_CASE("ComputationGraphAttrs to/from json") {
    ComputationGraphAttrs correct = BatchNormAttrs{true};
    json j = correct;
    auto result = j.get<ComputationGraphAttrs>();

    CHECK(result == correct);
  }

  TEST_CASE("PCGOperatorAttrs to/from json") {
    PCGOperatorAttrs correct = RepartitionAttrs{
      /*repartition_dim=*/ff_dim_t{1},
      /*repartition_degree=*/4,
    };
    json j = correct;
    auto result = j.get<PCGOperatorAttrs>();

    CHECK(result == correct);
  }
}
