#include "utils/fmt/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::pair<int, int>)") {
    std::pair<int, int> input = {3, 5};
    std::string result = fmt::to_string(input);
    std::string correct = "{3, 5}";
    CHECK(result == correct);
  }
}
