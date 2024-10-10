#include "utils/rapidcheck/variant.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Arbitrary<std::variant>") {
    RC_SUBCASE("valid type", [](std::variant<int, float> v) {
      return std::holds_alternative<int>(v) || std::holds_alternative<float>(v);
    });
  }
}
