#include "utils/fmt/optional.h"
#include "test/utils/doctest.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::optional<int>)") {
    SUBCASE("has value") {
      std::optional<int> input = 4;
      std::string result = fmt::to_string(input);
      std::string correct = "4";
      CHECK(result == correct);
    }

    SUBCASE("does not have value") {
      std::optional<int> input = std::nullopt;
      std::string result = fmt::to_string(input);
      std::string correct = "nullopt";
      CHECK(result == correct);
    }
  }
}
