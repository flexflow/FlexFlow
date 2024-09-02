#include "substitutions/operator_pattern/get_attribute.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_attribute(LinearAttrs, OperatorAttributeKey)") {
    int out_channels = 16;
    bool use_bias = true;
    std::optional<Activation> activation = Activation::GELU;
    std::optional<RegularizerAttrs> regularizer = RegularizerAttrs{
        L1RegularizerAttrs{
            0.5,
        },
    };

    LinearAttrs attrs = LinearAttrs{
        out_channels,
        use_bias,
        DataType::FLOAT,
        activation,
        regularizer,
    };

    SUBCASE("USE_BIAS") {
      std::optional<OperatorAttributeValue> result =
          get_attribute(attrs, OperatorAttributeKey::USE_BIAS);
      std::optional<OperatorAttributeValue> correct =
          OperatorAttributeValue{use_bias};
      CHECK(result == correct);
      CHECK(result.value().has<bool>());
    }
  }
}
