#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::vector<int>)") {
    std::vector<int> input = {0, 1, 3, 2};
    std::string result = fmt::to_string(input);
    std::string correct = "[0, 1, 3, 2]";
    CHECK(result == correct);
  }
}
