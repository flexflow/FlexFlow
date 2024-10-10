#include "utils/hash/tuple.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("std::hash<std::tuple<Ts...>>") {
    std::tuple<int, std::string, double> tuple1{1, "test", 3.14};
    std::tuple<int, std::string, double> tuple2{2, "test", 3.14};

    size_t hash1 = get_std_hash(tuple1);
    size_t hash2 = get_std_hash(tuple2);

    CHECK(hash1 != hash2);

    std::get<0>(tuple1) = 2;
    hash1 = get_std_hash(tuple1);
    CHECK(hash1 == hash2);
  }
}
