#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::unordered_set<T>>") {
    std::unordered_set<int> set1{1, 2, 3};
    std::unordered_set<int> set2{1, 2, 3, 4};

    size_t hash1 = get_std_hash(set1);
    size_t hash2 = get_std_hash(set2);

    CHECK(hash1 != hash2);

    set1.insert(4);
    hash1 = get_std_hash(set1);
    CHECK(hash1 == hash2);
  }
}
