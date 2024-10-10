#include "utils/containers/index_of.h"
#include "test/utils/doctest/fmt/optional.h"
#include <doctest/doctest.h>
#include <optional>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("index_of") {

    std::vector<int> v = {1, 2, 3, 4, 3, 5};

    SUBCASE("element occurs once in container") {
      CHECK(index_of(v, 4).value() == 3);
    }
    SUBCASE("if element appears multiple times, return the first occurrence") {
      CHECK(index_of(v, 3).value() == 2);
    }
    SUBCASE("element not in container") {
      CHECK(index_of(v, 7) == std::nullopt);
    }
  }
}
