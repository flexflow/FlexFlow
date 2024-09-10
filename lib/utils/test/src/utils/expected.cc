#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("optional_from_expected(tl::expected<T, E>)") {
    SUBCASE("has value") {
      tl::expected<int, std::string> input = 1;

      std::optional<int> result = optional_from_expected(input);
      std::optional<int> correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("has unexpected") {
      tl::expected<int, std::string> input =
          tl::make_unexpected("error message");

      std::optional<int> result = optional_from_expected(input);
      std::optional<int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
