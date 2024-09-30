#include "utils/containers/flatmap.h"
#include <doctest/doctest.h>
#include <string>
#include "test/utils/doctest/fmt/unordered_set.h"

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
}
