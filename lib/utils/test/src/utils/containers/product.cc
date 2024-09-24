#include "utils/containers/product.h"
#include <doctest/doctest.h>
#include <set>
#include <unordered_set>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE_TEMPLATE("product",
                     C,
                     std::vector<int>,
                     std::vector<double>,
                     std::set<int>,
                     std::unordered_set<int>) {

    SUBCASE("non-empty container") {
      C input = {1, -2, 3, 5};
      auto correct = -30;
      auto result = product(input);
      CHECK(correct == result);
    }

    SUBCASE("empty container") {
      C input = {};
      auto correct = 1;
      auto result = product(input);
      CHECK(correct == result);
    }
  }
}
