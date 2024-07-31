#include "utils/containers/transform.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform(std::vector<In>, F)") {
    std::vector<int> input = {1, 2, 3};
    std::vector<std::string> result =
        transform(input, [](int x) { return std::to_string(x); });
    std::vector<std::string> correct = {"1", "2", "3"};
    CHECK(result == correct);
  }

  TEST_CASE("transform(std::unordered_set<In>, F)") {
    std::unordered_set<int> input = {1, 2, 3};
    std::unordered_set<std::string> result =
        transform(input, [](int x) { return std::to_string(x); });
    std::unordered_set<std::string> correct = {"1", "2", "3"};
    CHECK(result == correct);
  }

  TEST_CASE("transform(std::string, F)") {
    std::string input = "abc";
    std::string result = transform(input, [](char x) -> char { return x + 1; });
    std::string correct = "bcd";
    CHECK(result == correct);
  }
}
