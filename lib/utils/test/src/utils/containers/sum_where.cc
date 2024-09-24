#include "utils/containers/sum_where.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("sum_where") {
    SUBCASE("Resulting container is non-empty") {
      std::vector<int> input = {1, 2, 3, 4, 5};
      auto condition = [](int x) { return x % 2 == 0; };
      int correct = 6;
      int result = sum_where(input, condition);
      CHECK(correct == result);
    }

    SUBCASE("Resulting container is empty") {
      std::vector<int> input = {1, 2, 3, 4, 5};
      auto condition = [](int x) { return x > 10; };
      int correct = 0;
      int result = sum_where(input, condition);
      CHECK(correct == result);
    }
  }
}
