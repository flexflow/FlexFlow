#include "utils/containers/value_all.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <optional>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("value_all") {
    SUBCASE("With nullopt") {
      std::vector<std::optional<int>> input = {1, 2, std::nullopt, 4, 5};
      CHECK_THROWS(value_all(input));
    }

    SUBCASE("Without nullopt") {
      std::vector<std::optional<int>> input = {1, 2, 3, 4, 5};
      std::vector<int> correct = {1, 2, 3, 4, 5};
      std::vector<int> result = value_all(input);
      CHECK(correct == result);
    }
  }
}
