#include "utils/containers/permutations.h"
#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("permutations") {
    SUBCASE("size=1") {
      std::vector<int> vec = {1};
      auto result = permutations(vec);
      std::unordered_set<std::vector<int>> correct = {{1}};
      CHECK(result == correct);
    }

    SUBCASE("size=3") {
      std::vector<int> vec = {1, 2, 3};
      auto result = permutations(vec);
      std::unordered_set<std::vector<int>> correct = {
          {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}};
      CHECK(result == correct);
    }

    SUBCASE("elements repeated") {
      std::vector<int> vec = {1, 2, 2};
      auto result = permutations(vec);
      std::unordered_set<std::vector<int>> correct = {
          {1, 2, 2}, {2, 1, 2}, {2, 2, 1}};
      CHECK(result == correct);
    }
  }
}
