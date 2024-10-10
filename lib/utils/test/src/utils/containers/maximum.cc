#include "utils/containers/maximum.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("maximum") {

    SUBCASE("non-empty container") {
      std::vector<int> input = {1, 5, 3, 4, 2};
      int correct = 5;
      int result = maximum(input);
      CHECK(correct == result);
    }

    SUBCASE("empty container") {
      std::vector<int> input = {};

      CHECK_THROWS(maximum(input));
    }
  }
}
