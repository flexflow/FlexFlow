#include "utils/hash_extra/hash_combine.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("hash_combine(std::size_t &, T const &)") {
    int x = 1;
    std::string y = "a";

    size_t h1 = 0;
    hash_combine(h1, x);
    hash_combine(h1, y);

    size_t h2 = 0;
    hash_combine(h2, y);
    hash_combine(h2, x);

    size_t h3 = 0;
    hash_combine(h3, x);
    hash_combine(h3, x);

    size_t h4 = 0;
    hash_combine(h4, y);
    hash_combine(h4, y);

    std::unordered_set<size_t> hashes = {h1, h2, h3, h4};
    CHECK(hashes.size() == 4); // all hashes should be unique
  }

  TEST_CASE("hash_combine(std::size_t &, T const &v, Ts... rest)") {
    int x = 1;
    std::string y = "a";
    std::string z = "bb";

    size_t result = 0;
    hash_combine(result, x, y, z);

    size_t correct = 0;
    hash_combine(correct, x);
    hash_combine(correct, y);
    hash_combine(correct, z);

    CHECK(result == correct);
  }
}
