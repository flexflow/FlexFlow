#include "utils/containers/index_of.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>
#include <optional>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("index_of") {

    std::vector<int> v = {1, 2, 3, 4, 3, 5};

    SUBCASE("unique element") {
      CHECK(index_of(v, 4).value() == 3);
    }
    SUBCASE("duplicate elements") {
      CHECK(index_of(v, 3).value() == 2); // Returns first occurrence
    }
    SUBCASE("element not present") {
      CHECK(index_of(v, 7) == std::nullopt);
    }
  }
}
