#include <doctest/doctest.h>
#include "utils/containers/intersection.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("intersection(std::unordered_set<T>, std::unordered_set<T>)") {
    std::unordered_set<int> input_l = {1, 2, 3};
    std::unordered_set<int> input_r = {2, 3, 5};

    std::unordered_set<int> result = intersection(input_l, input_r);
    std::unordered_set<int> correct = { 2, 3 };

    CHECK(result == correct);
  }

  TEST_CASE("intersection(C)") {
    SUBCASE("input is empty container") {
      std::vector<std::unordered_set<int>> input = {};
      
      std::optional<std::unordered_set<int>> result = intersection(input);
      std::optional<std::unordered_set<int>> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input is has only one set") {
      std::vector<std::unordered_set<int>> input = {{1, 2, 3}};
      
      std::optional<std::unordered_set<int>> result = intersection(input);
      std::optional<std::unordered_set<int>> correct = {{1, 2, 3}};

      CHECK(result == correct);
    }

    SUBCASE("input has multiple sets") {
      std::vector<std::unordered_set<int>> input = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
      
      std::optional<std::unordered_set<int>> result = intersection(input);
      std::optional<std::unordered_set<int>> correct = {{3}};

      CHECK(result == correct);
    }
  }
}
