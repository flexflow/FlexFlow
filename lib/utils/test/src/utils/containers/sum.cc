#include "utils/containers/sum.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("sum(std::vector<int>)") {
    SUBCASE("input is empty") {
      std::vector<int> input = {};

      int result = sum(input);
      int correct = 0;

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      std::vector<int> input = {1, 3, 2};

      int result = sum(input);
      int correct = 6;

      CHECK(result == correct);
    }
  }
}
