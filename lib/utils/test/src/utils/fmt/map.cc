#include "utils/fmt/map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::map<int, int>)") {
    std::map<int, int> input = {{0, 10}, {1, 1}, {3, 5}, {2, 8}};
    std::string result = fmt::to_string(input);
    std::string correct = "{{0, 10}, {1, 1}, {2, 8}, {3, 5}}";
    CHECK(result == correct);
  }
}
