#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BatchNormAttrs to/from json") {
    BatchNormAttrs correct = BatchNormAttrs{
      /*relu=*/false,
      /*affine=*/true,
      /*eps=*/1e-5,
      /*momentum=*/0.1,
    };

    nlohmann::json j = correct;
    BatchNormAttrs result = j.get<BatchNormAttrs>();

    CHECK(result == correct);
  }
}
