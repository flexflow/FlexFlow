#include "utils/containers/index_of.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>
#include <optional>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("index_of") {
    SUBCASE("unique elements") {
      std::vector<int> v = {1, 2, 3, 4, 5};
      CHECK(index_of(v, 3).value() == 2);
      CHECK(index_of(v, 6) == std::nullopt);
    }
    SUBCASE("duplicate elements") {
      std::vector<int> v = {1, 2, 3, 4, 3, 5};
      CHECK(index_of(v, 3).value() == 2);
      CHECK(index_of(v, 6) == std::nullopt);
    }
  }
}
