#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BatchNormAttrs to/from json") {
    BatchNormAttrs correct = BatchNormAttrs{true};

    nlohmann::json j = correct;
    BatchNormAttrs result = j.get<BatchNormAttrs>();

    CHECK(result == correct);
  }
}
