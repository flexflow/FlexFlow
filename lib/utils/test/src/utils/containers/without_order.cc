#include "utils/containers/without_order.h"
#include "utils/fmt/unordered_multiset.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("without_order") {
    std::vector<int> input = {1, 2, 3, 3, 2, 3};
    std::unordered_multiset<int> result = without_order(input);
    std::unordered_multiset<int> correct = {1, 2, 3, 3, 2, 3};
    CHECK(result == correct);
  }
}
