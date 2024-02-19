#include "utils/testing.h"
#include "utils/hash_extra/instances/unordered_set.h"
#include "utils/hash_extra/get_std_hash.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::unordered_set<T>>") {
    std::unordered_set<int> s1 = { 1, 2, 3, 4 };
    std::unordered_set<int> s2 = { 1, 2, 3 };

    CHECK(get_std_hash(s1) == get_std_hash(s1));
    CHECK(get_std_hash(s1) != get_std_hash(s2));
  }
}
