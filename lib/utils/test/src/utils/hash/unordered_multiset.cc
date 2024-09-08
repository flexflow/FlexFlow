#include "utils/hash/unordered_multiset.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::unordered_multiset<int>>") {
    std::unordered_multiset<int> input = {1, 2, 2, 1, 5};
    size_t input_hash = get_std_hash(input);

    SUBCASE("same values have the same hash") {
      std::unordered_multiset<int> also_input = {2, 1, 2, 5, 1};
      size_t also_input_hash = get_std_hash(input);

      CHECK(input_hash == also_input_hash);
    }

    SUBCASE("different values have different hashes") {
      SUBCASE("different number of duplicates") {
        std::unordered_multiset<int> other = {1, 2, 2, 1, 5, 5};
        size_t other_hash = get_std_hash(other);

        CHECK(input_hash != other_hash);
      }

      SUBCASE("different elements") {
        std::unordered_multiset<int> other = {1, 2, 2, 1, 6};
        size_t other_hash = get_std_hash(other);

        CHECK(input_hash != other_hash);
      }
    }
  }
}
