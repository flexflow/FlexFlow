#include "utils/testing.h"
#include "utils/algorithms/sorting.h"

TEST_CASE("Testing sorted_by function") {
  std::unordered_set<int> s = {5, 2, 3, 4, 1};
  auto sorted_s = sorted_by(s, [](int a, int b) { return a < b; });
  CHECK(sorted_s == std::vector<int>({1, 2, 3, 4, 5}));

  std::unordered_set<int> s2 = {-5, -1, -3, -2, -4};
  auto sorted_s2 = sorted_by(s2, [](int a, int b) { return a > b; });
  CHECK(sorted_s2 == std::vector<int>({-1, -2, -3, -4, -5}));
}

TEST_CASE("Testing compare_by function") {
  std::unordered_set<int> s = {5, 2, 3, 4, 1};
  std::vector<int> result =
      sorted_by(s, compare_by<int>([](int i) { return (-i); }));
  CHECK(result == std::vector<int>{5, 4, 3, 2, 1});
}
