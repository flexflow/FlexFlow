#include "utils/containers/enumerate.h"
#include <doctest/doctest.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("enumerate") {
    std::unordered_set<int> input_set = {1, 2, 3, 4, 5};
    std::unordered_map<size_t, int> result = enumerate(input_set);
    std::unordered_map<size_t, int> expected = {
        {1, 4}, {2, 3}, {3, 2}, {4, 1}, {0, 5}};
    CHECK(result == expected);
  }
}
