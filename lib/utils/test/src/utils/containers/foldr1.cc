#include "utils/containers/foldr1.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("foldr1(std::vector<T>, F)") {
    auto concat = [](std::string const &accum, std::string const &s) {
      return accum + s;
    };

    SUBCASE("empty input") {
      std::vector<std::string> input = {};
      CHECK_THROWS(foldr1(input, concat));
    }

    SUBCASE("non-empty input") {
      std::vector<std::string> input = {"ing", "tr", "a s"};

      std::string result = foldr1(input, concat);

      std::string correct = "a string";

      CHECK(result == correct);
    }
  }
}
