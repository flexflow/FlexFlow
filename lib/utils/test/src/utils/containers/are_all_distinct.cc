#include "utils/containers/are_all_distinct.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("are_all_distinct") {

    SUBCASE("Empty Container") {
      std::vector<int> input = {};
      CHECK(are_all_distinct(input));
    }
    SUBCASE("All elements are distinct") {
      std::vector<int> input = {1, 2, 3, 4};
      CHECK(are_all_distinct(input));
    }

    SUBCASE("Not all elements are distinct") {
      std::vector<int> input = {2, 2, 3, 4};
      CHECK_FALSE(are_all_distinct(input));
    }
  }
}
