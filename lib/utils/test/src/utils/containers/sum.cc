#include "utils/containers/sum.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("sum") {
    SUBCASE("non-empty container") {
      std::vector<int> input = {1, 2, 3, 4, 5};
      int correct = 15;
      int result = sum(input);
      CHECK(correct == result);
    }
    SUBCASE("empty container") {
      std::unordered_set<int> input = {};
      int correct = 0;
      int result = sum(input);
      CHECK(correct == result);
    }
  }
}
