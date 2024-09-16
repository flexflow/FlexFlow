#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("PCGOperatorAttrs to/from json") {
    PCGOperatorAttrs correct = PCGOperatorAttrs{RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1},
        /*repartition_degree=*/4,
    }};
    nlohmann::json j = correct;
    auto result = j.get<PCGOperatorAttrs>();

    CHECK(result == correct);
  }
}
