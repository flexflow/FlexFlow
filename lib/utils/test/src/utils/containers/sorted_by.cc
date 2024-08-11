#include "utils/containers/sorted_by.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Testing sorted_by function") {
    std::unordered_set<int> s = {5, 2, 3, 4, 1};
    auto sorted_s = sorted_by(s, [](int a, int b) { return a < b; });
    CHECK(sorted_s == std::vector<int>({1, 2, 3, 4, 5}));

    std::unordered_set<int> s2 = {-5, -1, -3, -2, -4};
    auto sorted_s2 = sorted_by(s2, [](int a, int b) { return a > b; });
    CHECK(sorted_s2 == std::vector<int>({-1, -2, -3, -4, -5}));
  }
}
