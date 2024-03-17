#include "utils/hash_extra/instances/tuple.h"
#include "utils/hash_extra/get_std_hash.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("hash<std::tuple<...>>") {
    std::tuple<int, std::string, char> t1 = {1, "a", 'b'}, t2 = {2, "a", 'b'},
                                       t3 = {1, "A", 'b'}, t4 = {1, "a", 'c'};

    CHECK(get_std_hash(t1) == get_std_hash(t1));
    CHECK(get_std_hash(t1) != get_std_hash(t2));
    CHECK(get_std_hash(t1) != get_std_hash(t3));
    CHECK(get_std_hash(t1) != get_std_hash(t4));
  }
}
