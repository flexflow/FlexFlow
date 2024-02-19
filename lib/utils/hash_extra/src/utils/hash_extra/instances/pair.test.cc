#include "utils/testing.h"
#include "utils/hash_extra/instances/pair.h"
#include "utils/hash_extra/get_std_hash.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("hash<std::pair<L, R>>") {
    std::pair<int, std::string> p1 = { 4, "a" };
    std::pair<int, std::string> p2 = { 4, "b" };
    std::pair<int, std::string> p3 = { 3, "a" };

    CHECK(get_std_hash(p1) == get_std_hash(p1));
    CHECK(get_std_hash(p1) != get_std_hash(p2));
    CHECK(get_std_hash(p1) != get_std_hash(p3));
  }
}
