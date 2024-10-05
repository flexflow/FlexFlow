#include "utils/containers/get_all_permutations_with_repetition.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/hash/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_all_permutations_with_repetition") {
    SUBCASE("output vector has only one element") {
      std::vector<int> input = {1, 2, 3};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations_with_repetition(input, 1));
      std::unordered_multiset<std::vector<int>> correct = {
          {1},
          {2},
          {3},
      };

      CHECK(result == correct);
    }

    SUBCASE("input vector has only one element") {
      std::vector<int> input = {1};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations_with_repetition(input, 2));
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 1},
      };

      CHECK(result == correct);
    }

    SUBCASE("input, output vectors have more than 1 element") {
      std::vector<int> input = {1, 2};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations_with_repetition(input, 3));
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 1, 1},
          {1, 1, 2},
          {1, 2, 1},
          {1, 2, 2},
          {2, 1, 1},
          {2, 1, 2},
          {2, 2, 1},
          {2, 2, 2},
      };

      CHECK(result == correct);
    }

    SUBCASE("duplicate elements") {
      std::vector<int> input = {1, 2, 2};

      std::unordered_multiset<std::vector<int>> result =
          unordered_multiset_of(get_all_permutations_with_repetition(input, 2));
      std::unordered_multiset<std::vector<int>> correct = {{1, 1},
                                                           {1, 2},
                                                           {1, 2},
                                                           {2, 1},
                                                           {2, 1},
                                                           {2, 2},
                                                           {2, 2},
                                                           {2, 2},
                                                           {2, 2}};

      CHECK(result == correct);
    }
  }
}
