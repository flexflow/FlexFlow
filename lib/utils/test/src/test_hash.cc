#include <doctest/doctest.h>
#include "utils/hash-utils.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("hash:unordered_map") {
    std::unordered_map<int, int> map1{{1, 2}};
    std::unordered_map<int, int> map2{{1, 2}, {3, 4}};

    size_t hash1 = get_std_hash(map1);
    size_t hash2 = get_std_hash(map2);

    CHECK(hash1 != hash2);

    map1.insert({1, 2});
    hash1 = get_std_hash(map1);
    CHECK(hash1 == hash2);
  }
}
