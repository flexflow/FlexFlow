#include "utils/containers/unordered_set_of.h"
#include <doctest/doctest.h>
#include <vector>
#include "utils/fmt/unordered_set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_set_of") {
    std::vector<int> input = {1, 2, 3, 3, 2, 3};
    std::unordered_set<int> result = unordered_set_of(input);
    std::unordered_set<int> correct = {1, 2, 3};
    CHECK(result == correct);
  }
}
