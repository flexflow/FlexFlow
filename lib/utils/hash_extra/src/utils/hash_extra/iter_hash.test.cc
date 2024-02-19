#include "utils/testing.h"
#include "utils/hash_extra/iter_hash.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("iter_hash") {
    std::vector<int> 
      v = { 1, 2, 3 };

    std::size_t result1 = 0;
    iter_hash(result1, v.begin(), v.end());

    std::size_t result1_again = 0;
    iter_hash(result1_again, v.begin(), v.end());

    std::size_t result2 = 0;
    iter_hash(result2, ++v.begin(), v.end());

    CHECK(result1 == result1_again);
    CHECK(result1 != result2);
  }
}
