#include "utils/hash/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::vector<T>>") {
    std::vector<int> vec1{1, 2, 3};
    std::vector<int> vec2{1, 2, 3, 4};

    size_t hash1 = get_std_hash(vec1);
    size_t hash2 = get_std_hash(vec2);

    CHECK(hash1 != hash2);

    vec1.push_back(4);
    hash1 = get_std_hash(vec1);
    CHECK(hash1 == hash2);
  }
}
