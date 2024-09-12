#include "utils/containers/sorted_by.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("sorted_by") {
    SUBCASE("sort increasing") {
      std::unordered_set<int> s = {5, 2, 3, 4, 1};
      std::vector<int> result =
          sorted_by(s, [](int a, int b) { return a < b; });
      std::vector<int> correct = {1, 2, 3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("sort decreasing") {
      std::unordered_set<int> input = {-5, -1, -3, -2, -4};
      std::vector<int> result =
          sorted_by(input, [](int a, int b) { return a > b; });
      std::vector<int> correct = {-1, -2, -3, -4, -5};
      CHECK(result == correct);
    }
  }
}
