#include "utils/fmt/unordered_set.h"
#include "test/utils/doctest.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fmt::to_string(std::unordered_set<int>)") {
    std::unordered_set<int> input = {0, 1, 3, 2};
    std::string result = fmt::to_string(input);
    std::string correct = "{0, 1, 2, 3}";
    CHECK(result == correct);
  }
}
