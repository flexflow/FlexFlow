#include "utils/containers/get_all_permutations.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/hash/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_all_permutations") {
    std::vector<int> input = {2, 1, 3};

    SUBCASE("contains all permutations") {
      std::unordered_set<std::vector<int>> result =
          unordered_set_of(get_all_permutations(input));
      std::unordered_set<std::vector<int>> correct = {
          {1, 2, 3},
          {1, 3, 2},
          {2, 1, 3},
          {2, 3, 1},
          {3, 1, 2},
          {3, 2, 1},
      };

      CHECK(result == correct);
    }

    SUBCASE("does not repeat permutations") {
      std::vector<std::vector<int>> result =
          as_vector(get_all_permutations(input));
      CHECK(result.size() == 6);
    }
  }
}
