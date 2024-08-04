#include "utils/containers/as_unordered_set.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("without_order") {
    std::vector<int> input = {1, 2, 3, 3, 2, 3};
    std::unordered_set<int> result = as_unordered_set(input);
    std::unordered_set<int> correct = {1, 2, 3};
    CHECK(result == correct);
  }
}
