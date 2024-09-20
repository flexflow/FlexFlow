#include "utils/fmt/variant.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::variant<int, std::string>)") {
    SUBCASE("has int") {
      std::variant<int, std::string> input = 4;
      std::string result = fmt::to_string(input);
      std::string correct = "4";
      CHECK(result == correct);
    }

    SUBCASE("has string") {
      std::variant<int, std::string> input = "hello world";
      std::string result = fmt::to_string(input);
      std::string correct = "hello world";
      CHECK(result == correct);
    }
  }
}
