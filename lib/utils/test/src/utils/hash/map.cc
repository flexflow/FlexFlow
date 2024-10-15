#include "utils/hash/map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::map<K, V>>") {
    std::map<int, int> map1{{1, 2}};
    std::map<int, int> map2{{1, 2}, {3, 4}};

    size_t hash1 = get_std_hash(map1);
    size_t hash2 = get_std_hash(map2);

    CHECK(hash1 != hash2);

    map1.insert({3, 4});
    hash1 = get_std_hash(map1);
    CHECK(hash1 == hash2);
  }
}
