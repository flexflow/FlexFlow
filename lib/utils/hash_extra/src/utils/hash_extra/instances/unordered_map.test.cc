#include "utils/hash_extra/instances/unordered_map.h"
#include "utils/hash_extra/get_std_hash.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::unordered_map<K, V>>") {
    std::unordered_map<int, std::string> m1 = {{1, "a"}, {2, "b"}},
                                         m2 = {{1, "a"}},
                                         m3 = {{2, "a"}, {1, "b"}};

    CHECK(get_std_hash(m1) == get_std_hash(m1));
    CHECK(get_std_hash(m1) != get_std_hash(m2));
    CHECK(get_std_hash(m1) != get_std_hash(m3));
  }
}
