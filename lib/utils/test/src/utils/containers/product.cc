#include "utils/containers/product.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("product") {
    SUBCASE("non-empty container") {
      std::vector<int> input = {1, -2, 3, 5};
      int correct = -30;
      int result = product(input);
      CHECK(correct == result);
    }

    SUBCASE("empty container") {
      std::vector<int> input = {};
      int correct = 1;
      int result = product(input);
      CHECK(correct == result);
    }
  }

  TEST_CASE("product_where") {
    SUBCASE("non-empty filtered container") {
      std::vector<int> input = {1, -2, 3, 4, 5};
      auto condition = [](int x) { return x % 2 == 0; };
      int correct = -8;
      int result = product_where(input, condition);
      CHECK(correct == result);
    }
    SUBCASE("empty filtered container") {
      std::vector<int> input = {1, 2, 3, 4, 5};
      auto condition = [](int x) { return x > 10; };
      int correct = 1;
      int result = product_where(input, condition);
      CHECK(correct == result);
    }
  }
}
