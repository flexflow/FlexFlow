#include "utils/containers/compare_by.h"
#include "utils/fmt/vector.h"
#include <algorithm>
#include <doctest/doctest.h>
#include <string>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compare_by") {
    std::vector<std::string> input = {"abc", "a", "ab"};
    auto comp = compare_by<std::string>(
        [](std::string const &s) { return s.length(); });
    std::vector<std::string> correct = {"a", "ab", "abc"};
    std::sort(input.begin(), input.end(), comp);
    CHECK(correct == input);
  }
}
