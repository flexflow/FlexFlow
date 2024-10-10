#include "utils/containers/get_all_assignments.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_all_assignments") {
    SUBCASE("empty input") {
      std::unordered_map<std::string, std::unordered_set<int>> input = {};

      std::unordered_set<std::unordered_map<std::string, int>> result =
          get_all_assignments(input);
      std::unordered_set<std::unordered_map<std::string, int>> correct = {{}};

      CHECK(result == correct);
    }

    SUBCASE("non-empty input") {
      std::unordered_map<std::string, std::unordered_set<int>> input = {
          {"a", {1, 2, 3}},
          {"b", {2, 3}},
      };

      std::unordered_set<std::unordered_map<std::string, int>> result =
          get_all_assignments(input);
      std::unordered_set<std::unordered_map<std::string, int>> correct = {
          {{"a", 1}, {"b", 2}},
          {{"a", 1}, {"b", 3}},
          {{"a", 2}, {"b", 2}},
          {{"a", 2}, {"b", 3}},
          {{"a", 3}, {"b", 2}},
          {{"a", 3}, {"b", 3}},
      };

      CHECK(result == correct);
    }

    SUBCASE("one possible-values set is empty") {
      std::unordered_map<std::string, std::unordered_set<int>> input = {
          {"a", {}},
          {"b", {2, 3}},
      };

      std::unordered_set<std::unordered_map<std::string, int>> result =
          get_all_assignments(input);
      std::unordered_set<std::unordered_map<std::string, int>> correct = {};

      CHECK(result == correct);
    }
  }
}
