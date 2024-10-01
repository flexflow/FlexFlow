#include "utils/containers/unordered_multiset_of.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_multiset_of") {
    std::vector<int> input = {1, 2, 3, 3, 2, 3};
    std::unordered_multiset<int> result = unordered_multiset_of(input);
    std::unordered_multiset<int> correct = {1, 2, 3, 3, 2, 3};
    CHECK(result == correct);
  }
}
