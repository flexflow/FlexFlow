#include "utils/hash_extra/get_std_hash.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_std_hash") {
    int x = 5;
    CHECK(get_std_hash(x) == std::hash<int>{}(x));
  }
}
