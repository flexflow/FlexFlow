#include "utils/containers/get_all_permutations.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/hash/vector.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/vector.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_all_permutations") {
    SUBCASE("input size 1") {
      std::vector<int> input = {1};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations(input));
      std::unordered_multiset<std::vector<int>> correct = {{1}};

      CHECK(result == correct);
    }

    SUBCASE("input size 3") {
      std::vector<int> input = {2, 1, 3};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations(input));
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 2, 3},
          {1, 3, 2},
          {2, 1, 3},
          {2, 3, 1},
          {3, 1, 2},
          {3, 2, 1},
      };

      CHECK(result == correct);
    }

    SUBCASE("elements repeated") {
      std::vector<int> input = {1, 2, 2};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations(input));
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 2, 2},
          {2, 1, 2},
          {2, 2, 1},
      };

      CHECK(result == correct);
    }
  }
}
