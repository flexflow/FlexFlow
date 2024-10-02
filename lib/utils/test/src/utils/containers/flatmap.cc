#include "utils/containers/flatmap.h"
#include <doctest/doctest.h>
#include <string>
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/containers/map_keys.h"
#include "utils/hash/pair.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatmap(std::unordered_set<T>, F)") {
    auto get_chars = [](std::string const &s) {
      std::unordered_set<char> result;
      for (char c : s) {
        result.insert(c);
      }
      return result;
    };

    SUBCASE("type changing") {
      std::unordered_set<std::string> input = {"hello", " ", "", "world", "!"};
      
      std::unordered_set<char> result = flatmap(input, get_chars);
      std::unordered_set<char> correct = {'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd', '!'};

      CHECK(result == correct);
    }

    SUBCASE("input is empty") {
      std::unordered_set<std::string> input = {};
      
      std::unordered_set<char> result = flatmap(input, get_chars);
      std::unordered_set<char> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("flatmap(std::unordered_map<K, V>, F)") {
    auto de_nest_keys = [](int k1, std::unordered_map<int, std::string> const &v) {
      return map_keys(v, [&](int k2) { return std::pair{k1, k2}; });
    };

    SUBCASE("input is empty") {
      std::unordered_map<int, std::unordered_map<int, std::string>> input = {};

      std::unordered_map<std::pair<int, int>, std::string> result = flatmap(input, de_nest_keys);
      std::unordered_map<std::pair<int, int>, std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      std::unordered_map<int, std::unordered_map<int, std::string>> input = {
        {
          1,
          {
            {2, "a"},
            {3, "b"},
          },
        },
        {
          2,
          {},
        },
        {
          3,
          {
            {3, "a"},
          },
        },
      };

      std::unordered_map<std::pair<int, int>, std::string> result = flatmap(input, de_nest_keys);
      std::unordered_map<std::pair<int, int>, std::string> correct = {
        {{1, 2}, "a"},
        {{1, 3}, "b"},
        {{3, 3}, "a"},
      };

      CHECK(result == correct);
    }

    SUBCASE("duplicate result keys") {
      auto always_return_same_map = [](int, std::string const &) {
        return std::unordered_map<std::string, int>{
          {"mykey", 10000},
        };
      };

      std::unordered_map<int, std::string> input = {
        {1, "a"},
        {2, "b"},
      };

      CHECK_THROWS(flatmap(input, always_return_same_map));
    }
  }
}
