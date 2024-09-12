#include "utils/containers/are_all_same.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("are_all_same") {
    SUBCASE("All elements are the same") {
      std::vector<int> input = {2, 2, 2, 2};
      CHECK(are_all_same(input));
    }

    SUBCASE("Not all elements are the same") {
      std::vector<int> input = {1, 2, 3, 4};
      CHECK_FALSE(are_all_same(input));
    }

    SUBCASE("Empty Container") {
      std::vector<int> input = {};
      CHECK_THROWS_AS(are_all_same(input), std::runtime_error);
    }
  }
}
