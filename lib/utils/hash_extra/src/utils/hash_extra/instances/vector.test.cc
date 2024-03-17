#include "utils/hash_extra/instances/vector.h"
#include "utils/hash_extra/get_std_hash.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::vector<T>>") {
    std::vector<int> v1 = {1, 2, 3}, v2 = {1, 2}, v3 = {3, 2, 1},
                     v4 = {1, 2, 4};

    CHECK(get_std_hash(v1) == get_std_hash(v1));
    CHECK(get_std_hash(v1) != get_std_hash(v2));
    CHECK(get_std_hash(v1) != get_std_hash(v3));
    CHECK(get_std_hash(v1) != get_std_hash(v4));
  }
}
