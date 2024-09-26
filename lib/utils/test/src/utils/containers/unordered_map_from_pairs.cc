#include "utils/containers/unordered_map_from_pairs.h"
#include <doctest/doctest.h>
#include <vector>
#include <string>
#include "utils/containers/contains.h"
#include "test/utils/doctest/fmt/unordered_map.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_map_from_pairs") {
    SUBCASE("nonempty input") {
      std::vector<std::pair<int, std::string>> input = {
        {1, "hello"},
        {3, "world"},
      };

      std::unordered_map<int, std::string> result = unordered_map_from_pairs(input);
      std::unordered_map<int, std::string> correct = {
        {1, "hello"},
        {3, "world"},
      };
      
      CHECK(result == correct);
    }

    SUBCASE("empty input") {
      std::vector<std::pair<int, std::string>> input = {};

      std::unordered_map<int, std::string> result = unordered_map_from_pairs(input);
      std::unordered_map<int, std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input with duplicate keys") {
      std::vector<std::pair<int, std::string>> input = {
        {1, "a"},
        {2, "c"},
        {1, "b"},
      };

      std::unordered_map<int, std::string> result = unordered_map_from_pairs(input);

      std::vector<std::unordered_map<int, std::string>> possible_correct_values = {
        {{1, "a"}, {2, "c"}},
        {{1, "b"}, {2, "c"}},
      };

      CHECK(contains(possible_correct_values, result));
    }
  }
}
